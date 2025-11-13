"""
DESCRIPTION

    Pure Python Backward-Chaining Knowledge Graph Generator.

WORKFLOW

    - Initialize BackwardChainer with all rules from the ontology parser.
    - Start from a "goal rule" and attempt to prove its conclusion.
        - Recursively find proofs for each premise of the rule.
        - Generate new individuals as needed for unbound variables.
        - Track recursive rule usage to prevent infinite loops.

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
    """

    def __init__(
        self,
        all_rules: List[ExecutableRule],
        max_recursion_depth: int = 2,
    ):
        """
        Initializes the chainer.

        Args:
            all_rules (List[ExecutableRule]):   All rules from the ontology parser.
            max_recursion_depth (int):          Max number of times a recursive rule can be used in a single proof path.
        """
        # Dict {rule_name: ExecutableRule}
        self.all_rules = {rule.name: rule for rule in all_rules}
        self.max_recursion_depth = max_recursion_depth

        # Initialize a Dict[Tuple, List[ExecutableRule]] for fast rule lookup based on the head
        self.rules_by_head = self._index_rules(all_rules)

        # Counters for generating unique entities
        self._var_rename_counter = 0
        self._individual_counter = 0

    def _get_atom_key(self, atom: Atom) -> Optional[Tuple]:
        """
        Creates a hashable key for an atom to index rules.

        Classes:    (rdf:type, 'Person')
        Relations:  ('hasParent', None)
        Attributes: ('age', None)
        """

        pred = atom.predicate
        obj = atom.object

        # Cannot index on a variable predicate
        if isinstance(pred, Var):
            return None

        # Key on (rdf:type, ClassName)
        if pred == RDF.type and not isinstance(obj, Var):
            if isinstance(obj, Class):
                # Class membership
                return (RDF.type, obj.name)
            else:
                print(f"Warning: Unexpected object for rdf:type: {obj}")
                return (RDF.type, str(obj))

        # Key on (PredicateName, None)
        else:
            if isinstance(pred, (Relation, Attribute)):
                return (pred.name, None)
            else:
                return (str(pred), None)

    def _index_rules(
        self, rules: List[ExecutableRule]
    ) -> Dict[Tuple, List[ExecutableRule]]:
        """
        Indexes rules by their conclusion (head atom) for faster lookup.

        Args:
            rules (List[ExecutableRule]): List of all rules.

        Returns:
            Dict[Tuple, List[ExecutableRule]]: Mapping from atom keys to rules.
        """
        index: Dict[Tuple, List[ExecutableRule]] = defaultdict(
            list
        )  # automatically creates lists on new keys

        # Index each rule by its conclusion
        for rule in rules:
            key = self._get_atom_key(rule.conclusion)
            if key is not None:
                index[key].append(rule)
            else:
                print(f"Warning: Cannot index rule with variable predicate: {rule}")

        # Return the completed index dictionary
        return index

    def _get_fresh_individual(self) -> Individual:
        """
        Generates a new, unique Individual.

        Returns:
            Individual: A new individual with a unique name.
        """

        idx = self._individual_counter
        self._individual_counter += 1

        return Individual(index=idx, name=f"Ind_{idx}")

    def _get_vars_in_atom(self, atom: Atom) -> Set[Var]:
        """
        Returns all Var objects present in an atom.
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
        Crete a new rule with all variables renamed to be unique.

        e.g., (X, P, Y) -> (X, Q, Z)  becomes
              (X_1, P, Y_1) -> (X_1, Q, Z_1)

        Above, the triples are Atoms and the tuple items are Terms.

        Args:
            rule (ExecutableRule): The rule to rename variables for.

        Returns:
            ExecutableRule: A new rule instance with renamed variables.
        """

        # Update the counter
        self._var_rename_counter += 1

        # Create a unique suffix
        suffix = f"_{self._var_rename_counter}"

        # A map from old Var to new Var
        var_map: Dict[Var, Var] = {}

        def get_renamed_var(v: Var) -> Var:
            # If the variable is not yet renamed, do so now
            if v not in var_map:
                var_map[v] = Var(v.name + suffix)

            # If it is already renamed, return that renamed Var
            return var_map[v]

        # Rename one Term (part of an Atom, ground or not (Var))
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
        Attempts to unify a GROUND goal atom with a rule's conclusion pattern.

        Args:
            goal:     A ground atom we want to prove  (e.g., parent(Ind_A, Ind_C))
            pattern:  A rule head with variables      (e.g., parent(X_1, Y_1))

        Returns:
            A substitution mapping from Var to Term if unification succeeds,
            or None if unification fails.
        """

        # Keep track of substitutions per Var, e.g.
        # { Var('X_1'): Individual('Ind_A'), ... }
        subst: Dict[Var, Term] = {}

        def unify_terms(t1: Term, t2: Term) -> bool:
            """
            Unify two terms, updating subst as needed.

            Args:
                t1: First term (ground)
                t2: Second term (could be Var or ground)

            Returns
                True on success, False on failure.
            """

            # If the second term t2 is a variable, try to bind it
            if isinstance(t2, Var):
                if t2 in subst and subst[t2] != t1:
                    # Variable already bound to different value
                    return False  # could not unify

                # Not already bound or same binding
                subst[t2] = t1
                # Successful binding
                return True

            # If the second term t2 is ground, then t1 must match it
            else:
                return t1 == t2

        # We assume that goal.subject, predicate, object are all ground
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

        # ------------------------------ RUNNING EXAMPLE ----------------------------- #
        #
        # Let's assume we want to prove the following rule:
        #       hasParent(X, Y) ∧ hasParent(Y, Z) → hasGrandparent(X, Z)
        #
        # This is expressed in owl 2 rl as:
        #       :hasGrandparent owl:propertyChainAxiom ( :hasParent :hasParent ) .
        #
        # Which is parsed into an ExecutableRule with:
        #       conclusion  =   Atom(Var('X'), hasGrandparent, Var('Z'))
        #       premises    = [ Atom(Var('X'), hasParent, Var('Y')),
        #                       Atom(Var('Y'), hasParent, Var('Z')) ]
        # ---------------------------------------------------------------------------- #

        # The 'raw', i.e. unrenamed, starting rule
        start_rule = self.all_rules[start_rule_name]
        # e.g. ExecutableRule(
        #   conclusion=Atom(Var('X'), hasGrandparent, Var('Z')),
        #   premises=[Atom(Var('X'), hasParent, Var('Y')), Atom(Var('Y'), hasParent, Var('Z'))]
        # )

        # Rename rule vars
        rule = self._rename_rule_vars(start_rule)
        # e.g. ExecutableRule(
        #   conclusion=Atom(Var('X_1'), hasGrandparent, Var('Z_1')),
        #   premises=[Atom(Var('X_1'), hasParent, Var('Y_1')), Atom(Var('Y_1'), hasParent, Var('Z_1'))]
        # )

        # Create initial substitution
        #   -> generate individuals for all variables in the rule's conclusion
        subst: Dict[Var, Term] = {}
        conclusion_vars = self._get_vars_in_atom(rule.conclusion)
        # e.g. X_1, Z_1

        if not conclusion_vars:
            # If the head has no variables, like Atom(Ind_A, hasParent, Ind_B),
            # it is already a fact and the generator can't start from it.
            print(
                f"Error: Rule '{start_rule_name}' conclusion has no variables. It is a fact."
            )
            return

        # Generate individuals for all conclusion variables
        for var in conclusion_vars:
            subst[var] = self._get_fresh_individual()
            # e.g. subst = {
            #           Var('X_1'): Individual('Ind_0'),
            #           Var('Z_1'): Individual('Ind_1')
            #      }

        # Create the ground goal we intend to prove
        ground_goal = rule.conclusion.substitute(subst)
        # e.g. Atom(Individual('Ind_0'), hasGrandparent, Individual('Ind_1'))

        # ------------------- Find all sub-proofs for the premises ------------------- #

        # This is the same logic as the recursive step in _find_proofs

        # A frozenset is a mutable unordered collection, good for tracking
        # recursive rule usage in proof paths like this.
        recursive_use_counts = frozenset()
        # The set will look like this:
        #   { ('rule_name_1', count_1), ('rule_name_2', count_2), ... }
        # Where the count is how many times that rule has been used in the current proof path.
        # If the count exceeds max_recursion_depth, we skip that rule.

        if rule.is_recursive():
            # If the starting rule is recursive, we count its first usage
            recursive_use_counts = frozenset([(rule.name, 1)])

        # Find unbound variables in premises and substitute them based on subst dictionary from above
        #   -> this new list can still contain variables that need grounding!
        premises_with_bound_vars = [p.substitute(subst) for p in rule.premises]
        # e.g. [ Atom(Individual('Ind_0'), hasParent, Var('Y_1')),
        #        Atom(Var('Y_1'), hasParent, Individual('Ind_1')) ]

        # Find any new variables (after the substitutions above) and add
        # them to the unbound variables set
        unbound_vars: Set[Var] = set()
        for p in premises_with_bound_vars:
            unbound_vars.update(self._get_vars_in_atom(p))
        # e.g. { Var('Y_1') }

        # Generate individuals for any unbound variables in the premises,
        # because before we try to prove these premises, they must be ground
        for var in unbound_vars:
            if var not in subst:
                # Should always be true
                # e.g. 'Y_1' not in subst = { 'X_1': Ind_0, 'Z_1': Ind_1 }
                subst[var] = self._get_fresh_individual()

        # e.g. subst = {
        #           Var('X_1'): Individual('Ind_0'),
        #           Var('Z_1'): Individual('Ind_1'),
        #           Var('Y_1'): Individual('Ind_2')     # new
        #      }

        # Final grounding of premises
        ground_premises = [p.substitute(subst) for p in premises_with_bound_vars]
        # e.g. [ Atom(Individual('Ind_0'), hasParent, Individual('Ind_2')),
        #        Atom(Individual('Ind_2'), hasParent, Individual('Ind_1')) ]

        # Keep track of all Iterator objects that are yielding sub-proofs for each premise
        premise_sub_proof_iters = []

        # Handle the case of zero-premise rules
        if not ground_premises:
            print("WARNING: Rule with no premises encountered.")

            # This rule has no premises, so it's a generator of base facts.
            yield Proof.create_base_proof(ground_goal)
            return

        failed_to_prove_a_premise = False

        # Find proofs for all (now ground) premises
        for premise in ground_premises:
            # Get all Iterator[Proof]'s for this premise
            # and pass the current recursion tracker
            proof_list = list(
                self._find_proofs_recursive(premise, recursive_use_counts)
            )

            if not proof_list:
                # No proofs found for this premise, so the whole rule fails
                failed_to_prove_a_premise = True
                break

            # Collect the list of proofs
            premise_sub_proof_iters.append(proof_list)

        # Check if any premise failed to find a proof
        if failed_to_prove_a_premise:
            return  # This whole proof for the goal fails

        # E.g., right now we have:
        #   premise_sub_proof_iters = [
        #       [Proof1_for_parent(Ind_0, Ind_2), Proof2_for_parent(Ind_0, Ind_2), ...],
        #       [Proof1_for_parent(Ind_2, Ind_1), Proof2_for_parent(Ind_2, Ind_1), ...],
        #   ]

        # Yield all combinations of sub-proofs by taking the Cartesian product of the iterators

        # e.g., for two premises A and B:
        #   for proof_A in proofs_for_A:
        #       for proof_B in proofs_for_B:
        #           yield Proof(goal, rule, [proof_A, proof_B])

        # for the grandparent example, we get that all_sub_proof_combinations will yield:
        #   (Proof1_for_parent(Ind_0, Ind_2), Proof1_for_parent(Ind_2, Ind_1))
        #   (Proof1_for_parent(Ind_0, Ind_2), Proof2_for_parent(Ind_2, Ind_1))
        #   (Proof2_for_parent(Ind_0, Ind_2), Proof1_for_parent(Ind_2, Ind_1))
        #   (Proof2_for_parent(Ind_0, Ind_2), Proof2_for_parent(Ind_2, Ind_1))
        for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
            yield Proof.create_derived_proof(
                goal=ground_goal,  # grounded goal atom
                rule=start_rule,  # unrenamed rule
                sub_proofs=list(sub_proof_combination),
            )

    def _find_proofs_recursive(
        self, goal_atom: Atom, recursive_use_counts: frozenset[Tuple[str, int]]
    ) -> Iterator[Proof]:
        """
        Recursively finds all possible proof trees for a given ground atom.

        Args:
            goal_atom (Atom):               A *ground* atom to prove (e.g., parent(Ind_A, Ind_C)).
            recursive_use_counts (frozenset):  Tracks recursive rule usage.

        Yields:
            Proof: A valid proof tree for the goal_atom.
        """

        # ------------------------------ RUNNING EXAMPLE ----------------------------- #
        #
        # Let's resume the running example of proving grandparent based on 2 x parent.
        #
        # Assume that in the generate_proof_trees function above, our goal was
        #       rule.conclusion = Atom(Var('X'), hasGrandparent, Var('Z'))
        # which got grounded to
        #       goal_atom = Atom(Individual('Ind_0'), hasGrandparent, Individual('Ind_1'))
        # leading - after creating individuals for the premises atoms - to the subst dictionary:
        #       subst = {
        #                   Var('X'): Individual('Ind_0'),
        #                   Var('Z'): Individual('Ind_1'),
        #                   Var('Y'): Individual('Ind_2')    # generated for the premises
        #               }
        # So, we now want to prove the premises:
        #       Atom(Individual('Ind_0'), hasParent, Individual('Ind_2'))
        #       Atom(Individual('Ind_2'), hasParent, Individual('Ind_1'))
        # That's where this recursive function comes in.
        #
        # Let's assume that the function is called for the first premise:
        #       goal_atom = Atom(Individual('Ind_0'), hasParent, Individual('Ind_2'))
        #
        # ---------------------------------------------------------------------------- #

        # Atom(Individual('Ind_0'), hasParent, Individual('Ind_2'))
        # gets converted to a hashable key for rule lookup:
        #   e.g. key = ('hasParent', None)
        key = self._get_atom_key(goal_atom)

        # Find all rules whose conclusion matches this atom,
        # i.e. rules we could apply to prove our goal_atom
        matching_rules = self.rules_by_head.get(key, [])
        #   e.g. [ ExecutableRule(  conclusion  =   Atom(Var('X'), hasParent, Var('Y')),
        #                           premises    = [ Atom(Var('Y'), hasChild, Var('X'))  ]),
        #          ... ]
        # These rules are not renamed yet!

        # ---------------------------------------------------------------------------- #
        #                                   BASE CASE                                  #
        # ---------------------------------------------------------------------------- #

        # If we can't use rules to prove the goal atom, it must be a base fact.
        if not matching_rules:
            yield Proof.create_base_proof(goal_atom)
            return

        # ---------------------------------------------------------------------------- #
        #                                RECURSIVE CASE                                #
        # ---------------------------------------------------------------------------- #

        # Try all matching rules to prove the goal atom
        for original_rule in matching_rules:
            #
            # If the rule is recursive
            #   -> check if we've hit max recursion depth
            #   -> update the recursion tracker
            if original_rule.is_recursive():
                #
                # Get the number of times this rule has been used recursively so far
                current_recursive_uses = dict(recursive_use_counts).get(
                    original_rule.name, 0
                )

                # Skip this rule if we've hit max recursion depth
                if current_recursive_uses >= self.max_recursion_depth:
                    continue

                # Make a new recursion tracker with updated count
                new_counts = dict(recursive_use_counts)
                new_counts[original_rule.name] = current_recursive_uses + 1
                new_recursive_use_counts = frozenset(new_counts.items())

            else:
                # Non-recursive rule, no changes to recursion tracker
                new_recursive_use_counts = recursive_use_counts

            # Rename the rule variables
            rule = self._rename_rule_vars(original_rule)
            # e.g. ExecutableRule(  conclusion  =   Atom(Var('X_3'), hasParent, Var('Y_3')),
            #                       premises    = [ Atom(Var('Y_3'), hasChild, Var('X_3'))  ])

            # Unify our ground goal with the rule conclusion (that has variables)
            #       Recall that _unify(t1, t2) returns a substitution dict from Var to Term
            #       and t1 needs to be GROUND.
            subst = self._unify(goal_atom, rule.conclusion)
            # e.g. unifying goal_atom       = Atom(Individual('Ind_0'), hasParent, Individual('Ind_2'))
            #      with     rule.conclusion = Atom(Var('X_3'), hasParent, Var('Y_3'))
            #      yields   subst = {
            #                   Var('X_3'): Individual('Ind_0'),
            #                   Var('Y_3'): Individual('Ind_2')
            #               }
            if subst is None:
                # Should not happen
                print(
                    f"Warning: Unification failed between goal {goal_atom} and rule conclusion {rule.conclusion}"
                )
                continue

            # Substitute the found substitutions into the rule premises
            premises_with_bound_vars = [p.substitute(subst) for p in rule.premises]
            # e.g. [ Atom(Individual('Ind_2'), hasChild, Individual('Ind_0')) ]

            # Find any new variables in the premises (after substitution)
            unbound_vars: Set[Var] = set()
            for p in premises_with_bound_vars:
                unbound_vars.update(self._get_vars_in_atom(p))

            # Generate individuals for any unbound variables in the premises
            for var in unbound_vars:
                if var not in subst:
                    # The if above should always be true
                    subst[var] = self._get_fresh_individual()

            # Final grounding of premises
            ground_premises = [p.substitute(subst) for p in premises_with_bound_vars]

            if not ground_premises:
                print("WARNING: Rule with no premises encountered.")
                # This is also a form of "base fact" for the chain.
                yield Proof.create_derived_proof(
                    goal=goal_atom, rule=original_rule, sub_proofs=[]
                )
                continue

            # Get all Iterators that yield the proofs for the premises
            # of the current rule
            premise_sub_proof_iters = []

            failed_to_prove_a_premise = False

            # Try to prove all premises
            for premise in ground_premises:
                # Recurse
                proof_list = list(
                    self._find_proofs_recursive(premise, recursive_use_counts)
                )

                if not proof_list:
                    # No proofs found for this premise, so the whole rule fails
                    failed_to_prove_a_premise = True
                    break

                # Collect the list of proofs
                premise_sub_proof_iters.append(proof_list)

            # Check if any premise failed to find a proof
            if failed_to_prove_a_premise:
                continue  # Try the next rule

            # Yield all combinations of sub-proofs by taking the Cartesian product of the iterators
            for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
                yield Proof.create_derived_proof(
                    goal=goal_atom,
                    rule=original_rule,
                    sub_proofs=list(sub_proof_combination),
                )
