"""
DESCRIPTION

    Pure Python Backward-Chaining Knowledge Graph Generator.

WORKFLOW

    - Initialize BackwardChainer with all rules from the ontology parser.
    - Start from a "goal rule" and attempt to prove its conclusion.
        - Recursively find proofs for each premise of the rule.
        - Generate new individuals as needed for unbound variables.
        - Track recursive rule usage to prevent infinite loops.
        - Track atoms in proof path to prevent ANY circular reasoning.

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
        verbose: bool = False,
    ):
        """
        Initializes the chainer.

        Args:
            all_rules (List[ExecutableRule]):   All rules from the ontology parser.
            max_recursion_depth (int):          Max number of times a recursive rule can be used in a single proof path.
            verbose (bool):                     Enable detailed debug output.
        """
        # Dict {rule_name: ExecutableRule}
        self.all_rules = {rule.name: rule for rule in all_rules}
        self.max_recursion_depth = max_recursion_depth
        self.verbose = verbose

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
        index: Dict[Tuple, List[ExecutableRule]] = defaultdict(list)

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
        visiting: Set[Tuple] = set()
        visited: Set[Tuple] = set()

        def dfs(key: Tuple):
            """Recursive DFS to find cycles."""
            visiting.add(key)

            for neighbor_key in graph.get(key, set()):
                if neighbor_key in visiting:
                    recursive_keys.add(key)
                elif neighbor_key not in visited:
                    dfs(neighbor_key)
                    if neighbor_key in recursive_keys:
                        recursive_keys.add(key)

            visiting.remove(key)
            visited.add(key)

        # Run DFS from every node to find all possible cycles
        all_keys = list(graph.keys())
        for key in all_keys:
            if key not in visited:
                dfs(key)

        # 3. Find all rule names whose conclusion is one of the recursive keys
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

        Args:
            rule (ExecutableRule): The rule to rename variables for.

        Returns:
            ExecutableRule: A new rule instance with renamed variables.
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
        subst: Dict[Var, Term] = {}

        def unify_terms(t1: Term, t2: Term) -> bool:
            if isinstance(t2, Var):
                if t2 in subst and subst[t2] != t1:
                    return False
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
            start_rule_name (str): The name of the rule to use as the starting point.

        Yields:
            Proof: A complete, ground proof tree.
        """
        if start_rule_name not in self.all_rules:
            print(f"Error: Rule '{start_rule_name}' not found.")
            return

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"GENERATING PROOF TREES FOR: {start_rule_name}")
            print(f"{'=' * 80}")

        start_rule = self.all_rules[start_rule_name]
        rule = self._rename_rule_vars(start_rule)

        # Create initial substitution
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

        if self.verbose:
            print(f"\nInitial substitution for conclusion variables:")
            for var, ind in subst.items():
                print(f"  {var} -> {ind}")

        ground_goal = rule.conclusion.substitute(subst)

        if self.verbose:
            print(f"\nGround goal to prove: {self._format_atom(ground_goal)}")

        # Track recursive rule usage
        recursive_use_counts = frozenset()
        if rule.name in self.recursive_rules:
            recursive_use_counts = frozenset([(rule.name, 1)])

        # Prepare premises
        premises_with_bound_vars = [p.substitute(subst) for p in rule.premises]
        unbound_vars: Set[Var] = set()
        for p in premises_with_bound_vars:
            unbound_vars.update(self._get_vars_in_atom(p))

        # Generate individuals for unbound variables
        for var in unbound_vars:
            if var not in subst:
                subst[var] = self._get_fresh_individual()
                if self.verbose:
                    print(f"  Generated {subst[var]} for unbound variable {var}")

        ground_premises = [p.substitute(subst) for p in premises_with_bound_vars]

        if not ground_premises:
            print("WARNING: Rule with no premises encountered.")
            yield Proof.create_base_proof(ground_goal)
            return

        if self.verbose:
            print(f"\nGround premises to prove:")
            for i, prem in enumerate(ground_premises):
                print(f"  {i + 1}. {self._format_atom(prem)}")

        atoms_in_path: frozenset[Atom] = frozenset([ground_goal])

        premise_sub_proof_iters = []
        failed_to_prove_a_premise = False

        # Find proofs for all premises
        for premise in ground_premises:
            proof_list = list(
                self._find_proofs_recursive(
                    premise, recursive_use_counts, atoms_in_path
                )
            )

            if not proof_list:
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

        # Yield all combinations of sub-proofs
        proof_count = 0
        for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
            proof_count += 1
            yield Proof.create_derived_proof(
                goal=ground_goal,
                rule=start_rule,
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

        Args:
            goal_atom (Atom):                   A *ground* atom to prove.
            recursive_use_counts (frozenset):   Tracks recursive rule usage.
            atoms_in_path (frozenset):          Atoms already in the current proof path
                                                 (prevents circular reasoning).

        Yields:
            Proof: A valid proof tree for the goal_atom.
        """
        if self.verbose:
            indent = "  " * len(atoms_in_path)
            print(f"{indent}→ Proving: {self._format_atom(goal_atom)}")

            if atoms_in_path and len(atoms_in_path) > 1:
                print(f"{indent}  Current proof path:")
                for i, atom in enumerate(atoms_in_path, 1):
                    print(f"{indent}    {i}. {self._format_atom(atom)}")

        # If yes, we cannot prove it at all (neither as base fact nor derived)
        if goal_atom in atoms_in_path:
            if self.verbose:
                print(
                    f"{indent}  ✗ CIRCULAR: Atom already in proof path, skipping entirely"
                )
            return  # Yield nothing - this would be circular

        # BASE CASE: Allow this as a base fact (only if not in path)
        if self.verbose:
            print(f"{indent}  ✓ Yielding BASE FACT proof")
        yield Proof.create_base_proof(goal_atom)

        # RECURSIVE CASE: Try to derive using rules
        # Add current atom to path BEFORE trying to derive it
        new_atoms_in_path = atoms_in_path | frozenset([goal_atom])

        key = self._get_atom_key(goal_atom)
        matching_rules = self.rules_by_head.get(key, [])

        if self.verbose and matching_rules:
            print(f"{indent}  Found {len(matching_rules)} matching rule(s)")

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

                new_counts = dict(recursive_use_counts)
                new_counts[original_rule.name] = current_recursive_uses + 1
                new_recursive_use_counts = frozenset(new_counts.items())
            else:
                new_recursive_use_counts = recursive_use_counts

            # Rename rule variables
            rule = self._rename_rule_vars(original_rule)

            # Get ALL variables in this renamed rule for proper debug output
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
                    if var in rule_vars:  # Only show variables from THIS rule
                        term_str = (
                            subst[var].name
                            if hasattr(subst[var], "name")
                            else str(subst[var])
                        )
                        print(f"{indent}      {var.name} → {term_str}")

            # Create a new dict for this rule's scope
            rule_subst = dict(subst)

            # Substitute into premises
            premises_with_bound_vars = [p.substitute(rule_subst) for p in rule.premises]
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

            ground_premises = [
                p.substitute(rule_subst) for p in premises_with_bound_vars
            ]

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
                    failed_to_prove_a_premise = True
                    if self.verbose:
                        print(
                            f"{indent}    ✗ Failed to prove premise: {self._format_atom(premise)}"
                        )
                    break

                premise_sub_proof_iters.append(proof_list)

            if failed_to_prove_a_premise:
                continue

            # Yield all combinations of sub-proofs
            for sub_proof_combination in itertools.product(*premise_sub_proof_iters):
                if self.verbose:
                    print(f"{indent}    ✓ Derived via {rule.name}")
                yield Proof.create_derived_proof(
                    goal=goal_atom,
                    rule=original_rule,
                    sub_proofs=list(sub_proof_combination),
                )

    def _format_atom(self, atom: Atom) -> str:
        """Helper to format atom for debug output."""
        s = atom.subject.name if hasattr(atom.subject, "name") else str(atom.subject)
        p = (
            atom.predicate.name
            if hasattr(atom.predicate, "name")
            else str(atom.predicate)
        )
        o = atom.object.name if hasattr(atom.object, "name") else str(atom.object)
        return f"({s}, {p}, {o})"
