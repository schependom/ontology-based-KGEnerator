"""
DESCRIPTION

    Pure Python Backward-Chaining Knowledge Graph Generator with detailed logging.

WORKFLOW

    - Initialize BackwardChainer with all rules from the ontology parser.
    - Start from a "goal rule" and attempt to prove its conclusion.
        - Recursively find proofs for each premise of the rule.
        - Generate new individuals as needed for unbound variables.
        - Track recursive rule usage to prevent infinite loops.
        - Track atoms in proof path to prevent circular reasoning.

AUTHOR

    Vincent Van Schependom
"""

from collections import defaultdict
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


class BackwardChainer:
    """
    Generates proof trees and synthetic data via backward chaining.

    This engine starts with a "goal rule" and tries to find all possible
    proof trees for its conclusion, generating new individuals along the way.

    Key features:
    - Backward chaining from goal to premises
    - Automatic individual generation for unbound variables
    - Recursion depth limiting for recursive rules
    - Circular reasoning prevention via atom path tracking
    - Optional verbose logging for debugging
    """

    def __init__(
        self,
        all_rules: List[ExecutableRule],
        max_recursion_depth: int = 2,
        verbose: bool = False,
    ):
        """
        Initializes the chainer.

        Args:
            all_rules (List[ExecutableRule]):   All rules from the ontology parser.
            max_recursion_depth (int):          Max number of times a recursive rule
                                                can be used in a single proof path.
            verbose (bool):                     Enable detailed debug output.
        """
        # Store rules as dict {rule_name: ExecutableRule} for fast lookup
        self.all_rules = {rule.name: rule for rule in all_rules}
        self.max_recursion_depth = max_recursion_depth
        self.verbose = verbose

        # Index rules by their conclusion for O(1) lookup
        # Dict[Tuple, List[ExecutableRule]]: atom_key -> [rules with that conclusion]
        self.rules_by_head = self._index_rules(all_rules)

        # Find all rule names that are part of a recursive cycle
        # (including mutual recursion, e.g., P -> Q, Q -> P)
        self.recursive_rules: Set[str] = self._find_recursive_rules(all_rules)
        if self.recursive_rules:
            print(f"Chainer identified {len(self.recursive_rules)} recursive rules:")
            for rule_name in self.recursive_rules:
                print(f"  - {rule_name}")

        # Counters for generating unique entities
        self._var_rename_counter = 0  # For renaming rule variables
        self._individual_counter = 0  # For generating new individuals

    def _get_atom_key(self, atom: Atom) -> Optional[Tuple]:
        """
        Creates a hashable key for an atom to index rules by their conclusions.

        Examples:
            Classes:    Atom(X, rdf:type, Person)     -> (rdf:type, 'Person')
            Relations:  Atom(X, hasParent, Y)         -> ('hasParent', None)
            Attributes: Atom(X, age, 25)              -> ('age', None)

        Args:
            atom (Atom): The atom to create a key for.

        Returns:
            Optional[Tuple]: A hashable key, or None if the predicate is a variable.
        """
        pred = atom.predicate
        obj = atom.object

        # Cannot index on a variable predicate
        if isinstance(pred, Var):
            return None

        # For class membership (rdf:type), key on (rdf:type, ClassName)
        if pred == RDF.type and not isinstance(obj, Var):
            if isinstance(obj, Class):
                return (RDF.type, obj.name)
            else:
                print(f"Warning: Unexpected object for rdf:type: {obj}")
                return (RDF.type, str(obj))

        # For relations and attributes, key on (PredicateName, None)
        else:
            if isinstance(pred, (Relation, Attribute)):
                return (pred.name, None)
            else:
                return (str(pred), None)

    def _index_rules(
        self, rules: List[ExecutableRule]
    ) -> Dict[Tuple, List[ExecutableRule]]:
        """
        Indexes rules by their conclusion (head atom) for O(1) lookup.

        This allows us to quickly find all rules that could prove a given atom.

        Args:
            rules (List[ExecutableRule]): List of all rules from the ontology.

        Returns:
            Dict[Tuple, List[ExecutableRule]]: Mapping from atom keys to rules
                                                 that have that atom as conclusion.
        """
        index: Dict[Tuple, List[ExecutableRule]] = defaultdict(list)

        for rule in rules:
            key = self._get_atom_key(rule.conclusion)
            if key is not None:
                index[key].append(rule)
            else:
                print(f"Warning: Cannot index rule with variable predicate: {rule}")

        return index

    def _find_recursive_rules(self, all_rules: List[ExecutableRule]) -> Set[str]:
        """
        Finds all rules that are part of any recursive cycle.

        This includes direct recursion (A -> A) and mutual recursion (A -> B, B -> A).
        Uses DFS to detect cycles in the dependency graph.

        Args:
            all_rules (List[ExecutableRule]): All rules from the ontology.

        Returns:
            Set[str]: Set of rule names that are part of recursive cycles.
        """
        # Build a dependency graph: atom_key -> {atom_keys it depends on}
        graph: Dict[Tuple, Set[Tuple]] = defaultdict(set)
        for rule in all_rules:
            head_key = self._get_atom_key(rule.conclusion)
            if head_key is None:
                continue
            for premise in rule.premises:
                premise_key = self._get_atom_key(premise)
                if premise_key is not None:
                    graph[head_key].add(premise_key)

        # Use DFS to find all nodes that are part of a cycle
        recursive_keys: Set[Tuple] = set()
        visiting: Set[Tuple] = set()  # Currently in recursion stack
        visited: Set[Tuple] = set()  # Fully explored

        def dfs(key: Tuple):
            """DFS to detect cycles and mark all nodes in cycles."""
            visiting.add(key)

            for neighbor_key in graph.get(key, set()):
                if neighbor_key in visiting:
                    # Back edge found - cycle detected!
                    recursive_keys.add(key)
                elif neighbor_key not in visited:
                    dfs(neighbor_key)
                    # Propagate "recursive" status up the call stack
                    if neighbor_key in recursive_keys:
                        recursive_keys.add(key)

            visiting.remove(key)
            visited.add(key)

        # Run DFS from every unvisited node
        for key in list(graph.keys()):
            if key not in visited:
                dfs(key)

        # Map recursive atom keys back to rule names
        recursive_rule_names: Set[str] = set()
        for rule in all_rules:
            head_key = self._get_atom_key(rule.conclusion)
            if head_key in recursive_keys:
                recursive_rule_names.add(rule.name)

        return recursive_rule_names

    def _get_fresh_individual(self) -> Individual:
        """
        Generates a new, unique Individual with an auto-incremented name.

        Returns:
            Individual: A new individual with a unique name (e.g., "Ind_0", "Ind_1").
        """
        idx = self._individual_counter
        self._individual_counter += 1
        return Individual(index=idx, name=f"Ind_{idx}")

    def _get_vars_in_atom(self, atom: Atom) -> Set[Var]:
        """
        Extracts all Var objects present in an atom.

        Args:
            atom (Atom): The atom to extract variables from.

        Returns:
            Set[Var]: Set of all variables in the atom.
        """
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
        Creates a new rule with all variables renamed to be unique.

        This prevents variable name collisions when using the same rule multiple times.

        Example:
            Input:  (X, parent, Y) -> (X, grandparent, Z)
            Output: (X_1, parent, Y_1) -> (X_1, grandparent, Z_1)

        Args:
            rule (ExecutableRule): The rule to rename variables for.

        Returns:
            ExecutableRule: A new rule instance with renamed variables.
        """
        self._var_rename_counter += 1
        suffix = f"_{self._var_rename_counter}"
        var_map: Dict[Var, Var] = {}

        def get_renamed_var(v: Var) -> Var:
            """Get or create a renamed version of a variable."""
            if v not in var_map:
                var_map[v] = Var(v.name + suffix)
            return var_map[v]

        def rename_term(t: Term) -> Term:
            """Rename a term if it's a variable, otherwise return as-is."""
            return get_renamed_var(t) if isinstance(t, Var) else t

        # Rename conclusion
        renamed_conclusion = Atom(
            subject=rename_term(rule.conclusion.subject),
            predicate=rename_term(rule.conclusion.predicate),
            object=rename_term(rule.conclusion.object),
        )

        # Rename all premises
        renamed_premises = [
            Atom(
                subject=rename_term(p.subject),
                predicate=rename_term(p.predicate),
                object=rename_term(p.object),
            )
            for p in rule.premises
        ]

        return ExecutableRule(
            name=rule.name, conclusion=renamed_conclusion, premises=renamed_premises
        )

    def _unify(self, goal: Atom, pattern: Atom) -> Optional[Dict[Var, Term]]:
        """
        Attempts to unify a GROUND goal atom with a rule's conclusion pattern.

        Unification finds a substitution that makes the pattern match the goal.

        Example:
            goal    = Atom(Ind_A, hasParent, Ind_C)  [ground]
            pattern = Atom(X_1, hasParent, Y_1)      [has variables]
            result  = {X_1: Ind_A, Y_1: Ind_C}       [substitution]

        Args:
            goal (Atom):    A ground atom we want to prove.
            pattern (Atom): A rule conclusion with variables.

        Returns:
            Optional[Dict[Var, Term]]: A substitution mapping from variables to
                                       ground terms, or None if unification fails.
        """
        subst: Dict[Var, Term] = {}

        def unify_terms(t1: Term, t2: Term) -> bool:
            """
            Unify two terms, updating the substitution dictionary.

            Args:
                t1 (Term): Ground term from goal.
                t2 (Term): Possibly variable term from pattern.

            Returns:
                bool: True if unification succeeds, False otherwise.
            """
            if isinstance(t2, Var):
                # t2 is a variable - try to bind it to t1
                if t2 in subst and subst[t2] != t1:
                    # Variable already bound to different value - unification fails
                    return False
                # Bind or confirm existing binding
                subst[t2] = t1
                return True
            else:
                # t2 is ground - must match t1 exactly
                return t1 == t2

        # Unify subject, predicate, and object
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

        This method generates all possible proof trees for the given rule's conclusion,
        creating new individuals as needed for variables.

        RUNNING EXAMPLE:
        ----------------
        Consider the rule for grandparent:
            hasParent(X, Y) ∧ hasParent(Y, Z) → hasGrandparent(X, Z)

        In OWL 2 RL:
            :hasGrandparent owl:propertyChainAxiom ( :hasParent :hasParent ) .

        Parsed as ExecutableRule:
            conclusion = Atom(Var('X'), hasGrandparent, Var('Z'))
            premises   = [Atom(Var('X'), hasParent, Var('Y')),
                         Atom(Var('Y'), hasParent, Var('Z'))]

        Args:
            start_rule_name (str): The name of the rule to use as starting point
                                   (e.g., "owl_chain_hasParent_hasParent_hasGrandparent").

        Yields:
            Proof: Complete, ground proof trees.
        """
        if start_rule_name not in self.all_rules:
            print(f"Error: Rule '{start_rule_name}' not found.")
            return

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"GENERATING PROOF TREES FOR: {start_rule_name}")
            print(f"{'=' * 80}")

        # Get the starting rule (unrenamed)
        start_rule = self.all_rules[start_rule_name]
        # Example: ExecutableRule(
        #   conclusion=Atom(Var('X'), hasGrandparent, Var('Z')),
        #   premises=[Atom(Var('X'), hasParent, Var('Y')),
        #             Atom(Var('Y'), hasParent, Var('Z'))]
        # )

        # Rename variables to avoid collisions
        rule = self._rename_rule_vars(start_rule)
        # Example after renaming: ExecutableRule(
        #   conclusion=Atom(Var('X_1'), hasGrandparent, Var('Z_1')),
        #   premises=[Atom(Var('X_1'), hasParent, Var('Y_1')),
        #             Atom(Var('Y_1'), hasParent, Var('Z_1'))]
        # )

        # Extract variables from the conclusion
        conclusion_vars = self._get_vars_in_atom(rule.conclusion)
        # Example: {Var('X_1'), Var('Z_1')}

        if not conclusion_vars:
            # If the conclusion has no variables, it's already a ground fact
            print(
                f"Error: Rule '{start_rule_name}' conclusion has no variables. It is a fact."
            )
            return

        # Generate fresh individuals for all conclusion variables
        subst: Dict[Var, Term] = {}
        for var in conclusion_vars:
            subst[var] = self._get_fresh_individual()
        # Example: subst = {
        #   Var('X_1'): Individual('Ind_0'),
        #   Var('Z_1'): Individual('Ind_1')
        # }

        if self.verbose:
            print(f"\nInitial substitution for conclusion variables:")
            for var, ind in subst.items():
                print(f"  {var} -> {ind}")

        # Create the ground goal we intend to prove
        ground_goal = rule.conclusion.substitute(subst)
        # Example: Atom(Individual('Ind_0'), hasGrandparent, Individual('Ind_1'))

        if self.verbose:
            print(f"\nGround goal to prove: {self._format_atom(ground_goal)}")

        # Track recursive rule usage
        recursive_use_counts = frozenset()
        if rule.name in self.recursive_rules:
            # If the starting rule is recursive, count its first usage
            recursive_use_counts = frozenset([(rule.name, 1)])

        # Substitute known variables into premises
        premises_with_bound_vars = [p.substitute(subst) for p in rule.premises]
        # Example: [Atom(Individual('Ind_0'), hasParent, Var('Y_1')),
        #           Atom(Var('Y_1'), hasParent, Individual('Ind_1'))]

        # Find any remaining unbound variables in premises
        unbound_vars: Set[Var] = set()
        for p in premises_with_bound_vars:
            unbound_vars.update(self._get_vars_in_atom(p))
        # Example: {Var('Y_1')}

        # Generate individuals for unbound variables
        for var in unbound_vars:
            if var not in subst:
                subst[var] = self._get_fresh_individual()
                if self.verbose:
                    print(f"  Generated {subst[var]} for unbound variable {var}")
        # Example: subst = {
        #   Var('X_1'): Individual('Ind_0'),
        #   Var('Z_1'): Individual('Ind_1'),
        #   Var('Y_1'): Individual('Ind_2')  # newly generated
        # }

        # Final grounding of premises
        ground_premises = [p.substitute(subst) for p in premises_with_bound_vars]
        # Example: [Atom(Individual('Ind_0'), hasParent, Individual('Ind_2')),
        #           Atom(Individual('Ind_2'), hasParent, Individual('Ind_1'))]

        # Handle zero-premise rules (axioms)
        if not ground_premises:
            print("WARNING: Rule with no premises encountered.")
            yield Proof.create_base_proof(ground_goal)
            return

        if self.verbose:
            print(f"\nGround premises to prove:")
            for i, prem in enumerate(ground_premises):
                print(f"  {i + 1}. {self._format_atom(prem)}")

        # Track atoms in the current proof path to prevent circular reasoning
        atoms_in_path: frozenset[Atom] = frozenset([ground_goal])

        # Find proofs for all premises
        premise_sub_proof_iters = []
        failed_to_prove_a_premise = False

        for premise in ground_premises:
            proof_list = list(
                self._find_proofs_recursive(
                    premise, recursive_use_counts, atoms_in_path
                )
            )

            if not proof_list:
                # No proofs found for this premise
                failed_to_prove_a_premise = True
                if self.verbose:
                    print(f"  ✗ Failed to prove: {self._format_atom(premise)}")
                break

            if self.verbose:
                print(
                    f"  ✓ Found {len(proof_list)} proof(s) for: {self._format_atom(premise)}"
                )
            premise_sub_proof_iters.append(proof_list)

        if failed_to_prove_a_premise:
            if self.verbose:
                print("\n✗ Proof generation failed (couldn't prove all premises)")
            return

        # Yield all combinations of sub-proofs (Cartesian product)
        # Example: If premise 1 has 2 proofs and premise 2 has 3 proofs,
        #          this yields 2 × 3 = 6 complete proof trees
        proof_count = 0
        for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
            proof_count += 1
            yield Proof.create_derived_proof(
                goal=ground_goal,
                rule=start_rule,  # Use unrenamed rule
                sub_proofs=list(sub_proof_combination),
            )

        if self.verbose:
            print(f"\n✓ Generated {proof_count} complete proof tree(s)")

    def _find_proofs_recursive(
        self,
        goal_atom: Atom,
        recursive_use_counts: frozenset[Tuple[str, int]],
        atoms_in_path: frozenset[Atom],
    ) -> Iterator[Proof]:
        """
        Recursively finds all possible proof trees for a given ground atom.

        RUNNING EXAMPLE (continued):
        ----------------------------
        From the generate_proof_trees example, we now need to prove the premises:
            1. Atom(Individual('Ind_0'), hasParent, Individual('Ind_2'))
            2. Atom(Individual('Ind_2'), hasParent, Individual('Ind_1'))

        Let's trace the first premise:
            goal_atom = Atom(Individual('Ind_0'), hasParent, Individual('Ind_2'))

        This gets converted to a key for rule lookup:
            key = ('hasParent', None)

        We find matching rules, e.g.:
            ExecutableRule(conclusion = Atom(Var('X'), hasParent, Var('Y')),
                          premises   = [Atom(Var('Y'), hasChild, Var('X'))])

        After renaming:
            ExecutableRule(conclusion = Atom(Var('X_3'), hasParent, Var('Y_3')),
                          premises   = [Atom(Var('Y_3'), hasChild, Var('X_3'))])

        Unifying goal_atom with the renamed conclusion:
            subst = {Var('X_3'): Individual('Ind_0'),
                    Var('Y_3'): Individual('Ind_2')}

        Substituting into premises:
            [Atom(Individual('Ind_2'), hasChild, Individual('Ind_0'))]

        Then we recursively prove this new premise...

        Args:
            goal_atom (Atom):               Ground atom to prove.
            recursive_use_counts (frozenset): Tracks how many times each recursive
                                              rule has been used in this path.
            atoms_in_path (frozenset):      Atoms already in the current proof path
                                            (prevents circular reasoning).

        Yields:
            Proof: Valid proof trees for the goal_atom.
        """
        if self.verbose:
            indent = "  " * len(atoms_in_path)
            print(f"{indent}→ Proving: {self._format_atom(goal_atom)}")

            if atoms_in_path and len(atoms_in_path) > 1:
                print(f"{indent}  Current proof path:")
                for i, atom in enumerate(atoms_in_path, 1):
                    print(f"{indent}    {i}. {self._format_atom(atom)}")

        # Check for circular reasoning
        if goal_atom in atoms_in_path:
            if self.verbose:
                print(
                    f"{indent}  ✗ CIRCULAR: Atom already in proof path, skipping entirely"
                )
            return  # Yield nothing - this would be circular

        # ==================== BASE CASE ==================== #
        # Allow this atom to be proven as a base fact
        if self.verbose:
            print(f"{indent}  ✓ Yielding BASE FACT proof")
        yield Proof.create_base_proof(goal_atom)

        # ==================== RECURSIVE CASE ==================== #
        # Try to derive using rules

        # Add current atom to path BEFORE trying to derive it
        new_atoms_in_path = atoms_in_path | frozenset([goal_atom])

        # Find rules that could prove this atom
        key = self._get_atom_key(goal_atom)
        matching_rules = self.rules_by_head.get(key, [])

        if self.verbose and matching_rules:
            print(f"{indent}  Found {len(matching_rules)} matching rule(s)")

        # Try each matching rule
        for original_rule in matching_rules:
            # Check recursion limits
            if original_rule.name in self.recursive_rules:
                current_recursive_uses = dict(recursive_use_counts).get(
                    original_rule.name, 0
                )

                if current_recursive_uses >= self.max_recursion_depth:
                    if self.verbose:
                        print(
                            f"{indent}  ✗ Skipping {original_rule.name} (recursion limit)"
                        )
                    continue

                # Update recursion counter
                new_counts = dict(recursive_use_counts)
                new_counts[original_rule.name] = current_recursive_uses + 1
                new_recursive_use_counts = frozenset(new_counts.items())
            else:
                # Non-recursive rule - no change to counter
                new_recursive_use_counts = recursive_use_counts

            # Rename rule variables to avoid collisions
            rule = self._rename_rule_vars(original_rule)

            # Get all variables in this renamed rule for debug output
            rule_vars = set()
            rule_vars.update(self._get_vars_in_atom(rule.conclusion))
            for premise in rule.premises:
                rule_vars.update(self._get_vars_in_atom(premise))

            if self.verbose:
                print(f"{indent}  Trying rule: {rule.name}")
                print(f"{indent}    Rule pattern: {self._format_atom(rule.conclusion)}")

            # Unify goal with rule conclusion
            subst = self._unify(goal_atom, rule.conclusion)
            if subst is None:
                if self.verbose:
                    print(f"{indent}    ✗ Unification failed")
                continue

            if self.verbose:
                print(f"{indent}    ✓ Unified with substitutions:")
                for var in sorted(subst.keys(), key=lambda v: v.name):
                    if var in rule_vars:  # Only show variables from this rule
                        term_str = (
                            subst[var].name
                            if hasattr(subst[var], "name")
                            else str(subst[var])
                        )
                        print(f"{indent}      {var.name} → {term_str}")

            # Create a new substitution dict for this rule's scope
            rule_subst = dict(subst)

            # Substitute into premises
            premises_with_bound_vars = [p.substitute(rule_subst) for p in rule.premises]

            # Find unbound variables
            unbound_vars: Set[Var] = set()
            for p in premises_with_bound_vars:
                unbound_vars.update(self._get_vars_in_atom(p))

            # Generate individuals for unbound variables
            newly_generated = []
            for var in unbound_vars:
                if var not in rule_subst:
                    rule_subst[var] = self._get_fresh_individual()
                    newly_generated.append((var, rule_subst[var]))

            if self.verbose and newly_generated:
                print(f"{indent}    Generated new individuals:")
                for var, ind in newly_generated:
                    print(f"{indent}      {var.name} → {ind.name}")

            # Final grounding of premises
            ground_premises = [
                p.substitute(rule_subst) for p in premises_with_bound_vars
            ]

            # Handle zero-premise rules (axioms)
            if not ground_premises:
                if self.verbose:
                    print(f"{indent}    ✓ No premises (axiom)")
                yield Proof.create_derived_proof(
                    goal=goal_atom, rule=original_rule, sub_proofs=[]
                )
                continue

            # Try to prove all premises (with updated atoms_in_path)
            premise_sub_proof_iters = []
            failed_to_prove_a_premise = False

            for premise in ground_premises:
                proof_list = list(
                    self._find_proofs_recursive(
                        premise, new_recursive_use_counts, new_atoms_in_path
                    )
                )

                if not proof_list:
                    # No proofs found for this premise
                    failed_to_prove_a_premise = True
                    if self.verbose:
                        print(
                            f"{indent}    ✗ Failed to prove premise: {self._format_atom(premise)}"
                        )
                    break

                premise_sub_proof_iters.append(proof_list)

            if failed_to_prove_a_premise:
                continue  # Try next rule

            # Yield all combinations of sub-proofs (Cartesian product)
            for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
                if self.verbose:
                    print(f"{indent}    ✓ Derived via {rule.name}")
                yield Proof.create_derived_proof(
                    goal=goal_atom,
                    rule=original_rule,  # Use unrenamed rule
                    sub_proofs=list(sub_proof_combination),
                )

    def _format_atom(self, atom: Atom) -> str:
        """
        Helper to format atom for debug output.

        Args:
            atom (Atom): The atom to format.

        Returns:
            str: A human-readable representation like "(Ind_0, hasParent, Ind_1)".
        """
        s = atom.subject.name if hasattr(atom.subject, "name") else str(atom.subject)
        p = (
            atom.predicate.name
            if hasattr(atom.predicate, "name")
            else str(atom.predicate)
        )
        o = atom.object.name if hasattr(atom.object, "name") else str(atom.object)
        return f"({s}, {p}, {o})"
