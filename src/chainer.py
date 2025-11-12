from data_structures import (
    Atom,
    ExecutableRule,
    Proof,
    Var,
    Term,
    Individual,
    Class,
    Relation,
    Attribute,
)
from rdflib.namespace import RDF
from typing import List, Dict, Set, Optional, Iterator, Tuple, Any
import itertools
import copy


class BackwardChainer:
    """
    Generates proof trees and synthetic data via backward chaining.

    This engine starts with a "goal rule" and tries to find all possible
    proof trees for its conclusion, generating new individuals along the way.
    """

    def __init__(
        self,
        all_rules: List[ExecutableRule],
        max_recursion_depth: int = 2,
    ):
        """
        Initializes the chainer.

        Args:
            all_rules (List[ExecutableRule]): All rules from the ontology parser.
            max_recursion_depth (int): Max number of times a recursive rule
                                       can be used in a single proof path.
        """
        self.max_recursion_depth = max_recursion_depth
        self.rules_by_head = self._index_rules(all_rules)
        self.all_rules = {rule.name: rule for rule in all_rules}

        # Counters for generating unique entities
        self._var_rename_counter = 0
        self._individual_counter = 0

    def _get_atom_key(self, atom: Atom) -> Optional[Tuple]:
        """
        Creates a hashable key for an atom to index rules.
        e.g., (rdf:type, 'Person') or ('hasParent', None)
        """
        pred = atom.predicate
        obj = atom.object

        if isinstance(pred, Var):
            # Cannot index on a variable predicate
            return None

        if pred == RDF.type and not isinstance(obj, Var):
            # Key on (rdf:type, ClassName)
            if isinstance(obj, Class):
                return (RDF.type, obj.name)
            else:
                return (RDF.type, str(obj))
        else:
            # Key on (PredicateName, None)
            if isinstance(pred, (Relation, Attribute)):
                return (pred.name, None)
            else:
                return (str(pred), None)

    def _index_rules(
        self, rules: List[ExecutableRule]
    ) -> Dict[Tuple, List[ExecutableRule]]:
        """
        Indexes rules by their conclusion (head atom) for faster lookup.
        """
        index: Dict[Tuple, List[ExecutableRule]] = {}
        for rule in rules:
            key = self._get_atom_key(rule.conclusion)
            if key:
                if key not in index:
                    index[key] = []
                index[key].append(rule)
        return index

    def _get_fresh_individual(self) -> Individual:
        """Generates a new, unique Individual."""
        idx = self._individual_counter
        self._individual_counter += 1
        # The index is arbitrary for generation, so -1 is fine
        # We use the counter for the name to ensure uniqueness
        return Individual(index=-1, name=f"Ind_{idx}")

    def _get_vars_in_atom(self, atom: Atom) -> Set[Var]:
        """Returns all Var objects present in an atom."""
        vars: Set[Var] = set()
        if isinstance(atom.subject, Var):
            vars.add(atom.subject)
        if isinstance(atom.predicate, Var):
            vars.add(atom.predicate)
        if isinstance(atom.object, Var):
            vars.add(atom.object)
        return vars

    def _rename_rule_vars(self, rule: ExecutableRule) -> ExecutableRule:
        """
        Returns a *new* rule with all variables renamed to be unique.
        e.g., (X, P, Y) -> (X, Q, Z)  becomes
              (X_1, P, Y_1) -> (X_1, Q, Z_1)
        """
        self._var_rename_counter += 1
        suffix = f"_{self._var_rename_counter}"

        var_map: Dict[Var, Var] = {}

        def get_renamed_var(v: Var) -> Var:
            if v not in var_map:
                var_map[v] = Var(v.name + suffix)
            return var_map[v]

        def rename_term(t: Term) -> Term:
            return get_renamed_var(t) if isinstance(t, Var) else t

        renamed_conclusion = Atom(
            subject=rename_term(rule.conclusion.subject),
            predicate=rename_term(rule.conclusion.predicate),
            object=rename_term(rule.conclusion.object),
        )
        renamed_premises = [
            Atom(
                subject=rename_term(p.subject),
                predicate=rename_term(p.predicate),
                object=rename_term(p.object),
            )
            for p in rule.premises
        ]

        # Return a new rule instance
        return ExecutableRule(
            name=rule.name, conclusion=renamed_conclusion, premises=renamed_premises
        )

    def _unify(self, goal: Atom, pattern: Atom) -> Optional[Dict[Var, Term]]:
        """
        Attempts to unify a goal atom with a rule's conclusion pattern.

        - goal: A ground atom we want to prove (e.g., parent(Ind_A, Ind_C))
        - pattern: A rule head with variables (e.g., parent(X_1, Y_1))

        Returns a substitution dict {Var -> Term} or None if no match.
        """
        subst: Dict[Var, Term] = {}

        def unify_terms(t1: Term, t2: Term) -> bool:
            """Unifies term t1 (from goal) with t2 (from pattern)."""
            if isinstance(t2, Var):
                if t2 in subst and subst[t2] != t1:
                    return False  # Variable already bound to different value
                subst[t2] = t1
                return True
            else:
                return t1 == t2

        if not unify_terms(goal.subject, pattern.subject):
            return None
        if not unify_terms(goal.predicate, pattern.predicate):
            return None
        if not unify_terms(goal.object, pattern.object):
            return None

        return subst

    def generate_proof_trees(self, start_rule_name: str) -> Iterator[Proof]:
        """
        Main entry point to generate proof trees starting from a specific rule.

        Args:
            start_rule_name (str): The name of the rule to use as the
                                   starting point (e.g., "transitive_hasAncestor").

        Yields:
            Proof: A complete, ground proof tree.
        """
        if start_rule_name not in self.all_rules:
            print(f"Error: Rule '{start_rule_name}' not found.")
            return

        start_rule = self.all_rules[start_rule_name]

        # 1. Rename rule to avoid collisions
        rule = self._rename_rule_vars(start_rule)

        # 2. Create initial substitution by generating individuals
        #    for all variables in the rule's conclusion.
        subst: Dict[Var, Term] = {}
        conclusion_vars = self._get_vars_in_atom(rule.conclusion)
        if not conclusion_vars:
            # If the head has no variables (e.t., rdf:type(A, owl:Class))
            # this generator can't start from it.
            print(f"Error: Rule '{start_rule_name}' conclusion has no variables.")
            return

        for var in conclusion_vars:
            subst[var] = self._get_fresh_individual()

        # 3. Create the ground goal we intend to prove
        ground_goal = rule.conclusion.substitute(subst)

        # 4. Find all sub-proofs for the premises
        # This is the same logic as the recursive step in _find_proofs

        # Check recursion
        recursion_tracker = frozenset()
        if rule.is_recursive():
            recursion_tracker = frozenset([(rule.name, 1)])

        # Find unbound variables in premises and ground them
        premise_sub_proof_iters = []
        bound_premises = [p.substitute(subst) for p in rule.premises]

        unbound_vars: Set[Var] = set()
        for p in bound_premises:
            unbound_vars.update(self._get_vars_in_atom(p))

        for var in unbound_vars:
            if var not in subst:  # Should always be true, but good check
                subst[var] = self._get_fresh_individual()

        final_premises = [p.substitute(subst) for p in bound_premises]

        if not final_premises:
            # This rule has no premises, so it's a generator of base facts.
            # This is rare in OWL but possible.
            # We treat its conclusion as a "base fact" in this context.
            yield Proof.create_base_proof(ground_goal)
            return

        # Find proofs for all (now ground) premises
        for premise in final_premises:
            # We pass the *original* rule's name/tracker
            iter_proofs = self._find_proofs_recursive(premise, recursion_tracker)
            premise_sub_proof_iters.append(list(iter_proofs))

        # Check if any premise failed to find a proof
        if any(not proofs for proofs in premise_sub_proof_iters):
            return  # This path fails

        # 5. Yield all combinations of sub-proofs
        for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
            yield Proof.create_derived_proof(
                goal=ground_goal,
                rule=start_rule,  # Use the original rule for the proof
                sub_proofs=list(sub_proof_combination),
            )

    def _find_proofs_recursive(
        self, goal_atom: Atom, recursion_tracker: frozenset[Tuple[str, int]]
    ) -> Iterator[Proof]:
        """
        Recursively finds all possible proof trees for a given ground atom.

        Args:
            goal_atom (Atom): A *ground* atom to prove (e.g., parent(Ind_A, Ind_C)).
            recursion_tracker (frozenset): Tracks recursive rule usage.

        Yields:
            Proof: A valid proof tree for the goal_atom.
        """

        key = self._get_atom_key(goal_atom)
        matching_rules = self.rules_by_head.get(key, [])

        # BASE CASE: No rules match this atom.
        # This atom is a "base fact" (a leaf) for this proof tree.
        if not matching_rules:
            yield Proof.create_base_proof(goal_atom)
            return

        # RECURSIVE STEP: Try every matching rule
        for original_rule in matching_rules:
            # Check recursion depth
            if original_rule.is_recursive():
                current_depth = dict(recursion_tracker).get(original_rule.name, 0)
                if current_depth >= self.max_recursion_depth:
                    continue  # Skip this rule

                # Update tracker for sub-proofs
                new_counts = dict(recursion_tracker)
                new_counts[original_rule.name] = current_depth + 1
                new_recursion_tracker = frozenset(new_counts.items())
            else:
                new_recursion_tracker = recursion_tracker

            # We must rename rule variables to avoid collision
            rule = self._rename_rule_vars(original_rule)

            # Unify our ground goal with the (variable) rule conclusion
            subst = self._unify(goal_atom, rule.conclusion)
            if subst is None:
                continue  # Should not happen if indexing is correct, but safe

            # Ground the premises using the substitution
            bound_premises = [p.substitute(subst) for p in rule.premises]

            # Find any new variables in the premises and ground them
            unbound_vars: Set[Var] = set()
            for p in bound_premises:
                unbound_vars.update(self._get_vars_in_atom(p))

            for var in unbound_vars:
                if (
                    var not in subst
                ):  # This is where 'Y' in parent(X,Y) gets instantiated
                    subst[var] = self._get_fresh_individual()

            final_premises = [p.substitute(subst) for p in bound_premises]

            if not final_premises:
                # Rule with no premises (e.g., A(x) -> B(x)).
                # This is also a form of "base fact" for the chain.
                # We yield a proof with this rule and no sub-proofs.
                yield Proof.create_derived_proof(
                    goal=goal_atom, rule=original_rule, sub_proofs=[]
                )
                continue

            # Find all proof combinations for the premises
            sub_proof_iters = []
            has_failed_premise = False
            for premise in final_premises:
                # Recurse
                proofs = list(
                    self._find_proofs_recursive(premise, new_recursion_tracker)
                )
                if not proofs:
                    has_failed_premise = True
                    break  # This premise is unprovable, so this rule fails
                sub_proof_iters.append(proofs)

            if has_failed_premise:
                continue  # Try the next rule

            # We have proofs for all premises. Yield all combinations.
            for sub_proof_combination in itertools.product(*sub_proof_iters):
                yield Proof.create_derived_proof(
                    goal=goal_atom,
                    rule=original_rule,
                    sub_proofs=list(sub_proof_combination),
                )


# ---------------------------------------------------------------------------- #
#                                 EXAMPLE USAGE                                #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # --- This is a mock-up of your parser output ---
    # Define Classes, Relations, etc.
    R_parent = Relation(0, "parent")
    R_grandparent = Relation(1, "grandparent")
    R_child = Relation(2, "child")

    # Define Rules from your example:
    # Rule1: parent(X,Y), parent(Y,Z) -> grandparent(X,Z)
    rule1 = ExecutableRule(
        name="rule_grandparent",
        conclusion=Atom(Var("X"), R_grandparent, Var("Z")),
        premises=[
            Atom(Var("X"), R_parent, Var("Y")),
            Atom(Var("Y"), R_parent, Var("Z")),
        ],
    )

    # Rule2: child(Y,X) -> parent(X,Y)
    rule2 = ExecutableRule(
        name="rule_parent_from_child",
        conclusion=Atom(Var("X"), R_parent, Var("Y")),
        premises=[Atom(Var("Y"), R_child, Var("X"))],
    )

    all_rules = [rule1, rule2]
    # --- End of mock-up ---

    print("Initializing Backward Chainer...")
    # We set max_recursion_depth to 1 (it's not a recursive example)
    chainer = BackwardChainer(all_rules=all_rules, max_recursion_depth=1)

    print("Generating proof trees starting from 'rule_grandparent'...")

    # We ask the chainer to generate proofs for "rule_grandparent"
    proof_generator = chainer.generate_proof_trees(start_rule_name="rule_grandparent")

    generated_proofs = []
    try:
        for i in range(5):  # Let's just get the first 5 proofs
            proof = next(proof_generator)
            generated_proofs.append(proof)
    except StopIteration:
        pass  # No more proofs to generate

    print(f"\n--- Generated {len(generated_proofs)} Proof Tree(s) ---")

    for i, proof in enumerate(generated_proofs):
        print(f"\n--- Proof {i + 1} for {proof.goal} ---")

        # Get the "minimal set of atoms"
        base_facts = proof.get_base_facts()
        print(f"  Base Facts ({len(base_facts)}):")
        for fact in base_facts:
            print(f"    - {fact}")

        print(f"  Rule: {proof.rule.name}")

        # You can traverse the tree, but here's a simple view
        print("  Sub-Proofs:")
        for sp in proof.sub_proofs:
            print(f"    - {sp.goal} (proven by {sp.rule.name})")
            for ssp in sp.sub_proofs:
                print(f"      - {ssp.goal} (proven by {ssp.rule.name})")
                # ... and so on
                # Base fact sub-proofs
                if not ssp.sub_proofs:
                    print(f"        - {ssp.goal} (Base Fact)")

    # Example Output:
    #
    # --- Proof 1 for Atom(subject=Individual(index=-1, name='Ind_0'), predicate=Relation(index=1, name='grandparent'), object=Individual(index=-1, name='Ind_1')) ---
    #   Base Facts (2):
    #     - Atom(subject=Individual(index=-1, name='Ind_2'), predicate=Relation(index=2, name='child'), object=Individual(index=-1, name='Ind_0'))
    #     - Atom(subject=Individual(index=-1, name='Ind_1'), predicate=Relation(index=2, name='child'), object=Individual(index=-1, name='Ind_2'))
    #   Rule: rule_grandparent
    #   Sub-Proofs:
    #     - Atom(subject=Individual(index=-1, name='Ind_0'), predicate=Relation(index=0, name='parent'), object=Individual(index=-1, name='Ind_2')) (proven by rule_parent_from_child)
    #       - Atom(subject=Individual(index=-1, name='Ind_2'), predicate=Relation(index=2, name='child'), object=Individual(index=-1, name='Ind_0')) (Base Fact)
    #     - Atom(subject=Individual(index=-1, name='Ind_2'), predicate=Relation(index=0, name='parent'), object=Individual(index=-1, name='Ind_1')) (proven by rule_parent_from_child)
    #       - Atom(subject=Individual(index=-1, name='Ind_1'), predicate=Relation(index=2, name='child'), object=Individual(index=-1, name='Ind_2')) (Base Fact)
