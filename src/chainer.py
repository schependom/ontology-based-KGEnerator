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
        max_proofs_per_rule: int = 100,  # Add limit to prevent explosion
    ):
        """
        Initializes the chainer.

        Args:
            all_rules (List[ExecutableRule]):   All rules from the ontology parser.
            max_recursion_depth (int):          Max number of times a recursive rule can be used in a single proof path.
            max_proofs_per_rule (int):          Max number of proof trees to generate per starting rule.
        """
        # Dict {rule_name: ExecutableRule}
        self.all_rules = {rule.name: rule for rule in all_rules}
        self.max_recursion_depth = max_recursion_depth
        self.max_proofs_per_rule = max_proofs_per_rule

        # Initialize a Dict[Tuple, List[ExecutableRule]] for fast rule lookup based on the head
        self.rules_by_head = self._index_rules(all_rules)

        # Find all rule names that are part of a recursive cycle
        self.recursive_rules: Set[str] = self._find_recursive_rules(all_rules)
        if self.recursive_rules:
            print(f"Chainer identified {len(self.recursive_rules)} recursive rules:")
            for rule_name in self.recursive_rules:
                print(f"  - {rule_name}")

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

    def _find_recursive_rules(self, all_rules: List[ExecutableRule]) -> Set[str]:
        """
        Finds all rules that are part of any recursive cycle (including mutual recursion).
        e.g., P -> Q, Q -> P
        """
        # 1. Build a dependency graph between "atom keys" (predicates/classes)
        #    graph: { head_key: Set(premise_key, ...) }
        graph: Dict[Tuple, Set[Tuple]] = defaultdict(set)
        for rule in all_rules:
            head_key = self._get_atom_key(rule.conclusion)
            if head_key is None:
                continue
            for premise in rule.premises:
                premise_key = self._get_atom_key(premise)
                if premise_key is not None:
                    graph[head_key].add(premise_key)

        # 2. Use DFS to find all nodes (atom keys) that are part of a cycle
        recursive_keys: Set[Tuple] = set()

        # We need to track nodes currently in the recursion stack (visiting)
        # and nodes we've already fully explored (visited)
        visiting: Set[Tuple] = set()
        visited: Set[Tuple] = set()

        def dfs(key: Tuple):
            """Recursive DFS to find cycles."""
            visiting.add(key)

            for neighbor_key in graph.get(key, set()):
                if neighbor_key in visiting:
                    # Cycle detected! This key is part of a cycle.
                    recursive_keys.add(key)
                elif neighbor_key not in visited:
                    dfs(neighbor_key)
                    # Propagate "recursive" status up the call stack
                    if neighbor_key in recursive_keys:
                        recursive_keys.add(key)

            visiting.remove(key)
            visited.add(key)

        # Run DFS from every node to find all possible cycles
        all_keys = list(graph.keys())
        for key in all_keys:
            if key not in visited:
                dfs(key)

        # 3. Find all rule names whose *conclusion* is one of the recursive keys
        recursive_rule_names: Set[str] = set()
        for rule in all_rules:
            head_key = self._get_atom_key(rule.conclusion)
            if head_key in recursive_keys:
                recursive_rule_names.add(rule.name)

        return recursive_rule_names

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
        Create a new rule with all variables renamed to be unique.

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

        start_rule = self.all_rules[start_rule_name]

        # Rename rule vars
        rule = self._rename_rule_vars(start_rule)

        # Create initial substitution
        #   -> generate individuals for all variables in the rule's conclusion
        subst: Dict[Var, Term] = {}
        conclusion_vars = self._get_vars_in_atom(rule.conclusion)

        if not conclusion_vars:
            print(
                f"Error: Rule '{start_rule_name}' conclusion has no variables. It is a fact."
            )
            return

        # Generate individuals for all conclusion variables
        for var in conclusion_vars:
            subst[var] = self._get_fresh_individual()

        # Create the ground goal we intend to prove
        ground_goal = rule.conclusion.substitute(subst)

        # Initialize recursion tracking
        recursive_use_counts = frozenset()
        if rule.name in self.recursive_rules:
            recursive_use_counts = frozenset([(rule.name, 1)])

        # Find unbound variables in premises and substitute them based on subst dictionary
        premises_with_bound_vars = [p.substitute(subst) for p in rule.premises]

        # Find any new variables (after the substitutions above)
        unbound_vars: Set[Var] = set()
        for p in premises_with_bound_vars:
            unbound_vars.update(self._get_vars_in_atom(p))

        # Generate individuals for any unbound variables in the premises
        for var in unbound_vars:
            if var not in subst:
                subst[var] = self._get_fresh_individual()

        # Final grounding of premises
        ground_premises = [p.substitute(subst) for p in premises_with_bound_vars]

        # Handle the case of zero-premise rules
        if not ground_premises:
            print(f"WARNING: Rule '{start_rule_name}' with no premises encountered.")
            yield Proof.create_base_proof(ground_goal)
            return

        # Find proofs for all (now ground) premises
        premise_sub_proof_iters = []
        failed_to_prove_a_premise = False

        for premise in ground_premises:
            proof_list = list(
                self._find_proofs_recursive(premise, recursive_use_counts)
            )

            if not proof_list:
                failed_to_prove_a_premise = True
                break

            premise_sub_proof_iters.append(proof_list)

        if failed_to_prove_a_premise:
            return

        # Limit the number of proofs generated
        proof_count = 0
        for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
            if proof_count >= self.max_proofs_per_rule:
                print(
                    f"  -> Reached max proofs limit ({self.max_proofs_per_rule}) for rule {start_rule_name}"
                )
                break

            yield Proof.create_derived_proof(
                goal=ground_goal,
                rule=start_rule,
                sub_proofs=list(sub_proof_combination),
            )
            proof_count += 1

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

        key = self._get_atom_key(goal_atom)
        matching_rules = self.rules_by_head.get(key, [])

        # ---------------------------------------------------------------------------- #
        #                                   BASE CASE                                  #
        # ---------------------------------------------------------------------------- #

        # Only yield base fact if no rules match OR if we want to allow base facts
        # For a pure synthetic generator, we should only yield base facts when
        # there are no applicable rules
        if not matching_rules:
            yield Proof.create_base_proof(goal_atom)
            return

        # ---------------------------------------------------------------------------- #
        #                                RECURSIVE CASE                                #
        # ---------------------------------------------------------------------------- #

        has_valid_proof = False

        # Try all matching rules to prove the goal atom
        for original_rule in matching_rules:
            # Check recursion limits
            if original_rule.name in self.recursive_rules:
                current_recursive_uses = dict(recursive_use_counts).get(
                    original_rule.name, 0
                )

                if current_recursive_uses >= self.max_recursion_depth:
                    continue

                new_counts = dict(recursive_use_counts)
                new_counts[original_rule.name] = current_recursive_uses + 1
                new_recursive_use_counts = frozenset(new_counts.items())
            else:
                new_recursive_use_counts = recursive_use_counts

            # Rename the rule variables
            rule = self._rename_rule_vars(original_rule)

            # Unify our ground goal with the rule conclusion
            subst = self._unify(goal_atom, rule.conclusion)
            if subst is None:
                continue

            # Substitute the found substitutions into the rule premises
            premises_with_bound_vars = [p.substitute(subst) for p in rule.premises]

            # Find any new variables in the premises (after substitution)
            unbound_vars: Set[Var] = set()
            for p in premises_with_bound_vars:
                unbound_vars.update(self._get_vars_in_atom(p))

            # Generate individuals for any unbound variables in the premises
            for var in unbound_vars:
                if var not in subst:
                    subst[var] = self._get_fresh_individual()

            # Final grounding of premises
            ground_premises = [p.substitute(subst) for p in premises_with_bound_vars]

            if not ground_premises:
                # Zero-premise rule
                has_valid_proof = True
                yield Proof.create_derived_proof(
                    goal=goal_atom, rule=original_rule, sub_proofs=[]
                )
                continue

            # Try to prove all premises
            premise_sub_proof_iters = []
            failed_to_prove_a_premise = False

            for premise in ground_premises:
                proof_list = list(
                    self._find_proofs_recursive(premise, new_recursive_use_counts)
                )

                if not proof_list:
                    failed_to_prove_a_premise = True
                    break

                premise_sub_proof_iters.append(proof_list)

            if failed_to_prove_a_premise:
                continue

            # Yield all combinations of sub-proofs
            has_valid_proof = True
            for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
                yield Proof.create_derived_proof(
                    goal=goal_atom,
                    rule=original_rule,
                    sub_proofs=list(sub_proof_combination),
                )

        # If no rules could prove this atom, yield it as a base fact
        if not has_valid_proof:
            yield Proof.create_base_proof(goal_atom)
