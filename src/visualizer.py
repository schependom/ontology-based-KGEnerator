"""
DESCRIPTION:

    Proof Tree Visualizer with Variable Substitutions and Constraint Checking.

AUTHOR:

    Vincent Van Schependom
"""

import argparse
import os
import sys
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import graphviz

# Custom imports
from data_structures import (
    Proof,
    Atom,
    ExecutableRule,
    Var,
    Individual,
    Class,
    Relation,
)

# REFACTORED: Import KGenerator from generator.py
from generator import KGenerator

from rdflib.namespace import RDF


class ProofTreeVisualizer:
    """
    Enhanced visualizer that tracks and displays variable substitutions
    throughout the proof tree, and validates proofs against constraints.

    This visualizer reconstructs the variable bindings at each step of the
    proof by matching rule patterns against ground instances, providing
    insight into how the backward chainer generates individuals and builds proofs.

    REFACTORED: Now uses KGenerator for proof generation.
    """

    def __init__(self, output_dir: str = "output"):
        """
        Initialize the visualizer.

        Args:
            output_dir (str): Directory to save output files. Created if it doesn't exist.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Tracking for graphviz node creation
        self.node_counter = 0  # Auto-increment for unique node IDs
        self.node_ids: Dict[Proof, str] = {}  # Map proof objects to node IDs

        # Variable substitution tracking
        # Maps each proof to the variable bindings that were used to derive it
        self.substitution_history: Dict[Proof, Dict[Var, str]] = {}

        # Constraint validation tracking
        # Maps each proof to whether it satisfies constraints
        self.constraint_validation: Dict[Proof, bool] = {}
        self.constraint_violations: Dict[Proof, List[str]] = {}

    def visualize_all_proofs(
        self,
        generator: KGenerator,
        rule_names: Optional[List[str]] = None,
        max_proofs_per_rule: int = 5,
        show_invalid_proofs: bool = False,
    ) -> None:
        """
        Generate visualizations for all rules (or specified rules).

        For each rule, generates up to max_proofs_per_rule proof trees,
        creating both text and graphical representations.

        REFACTORED: Now uses KGenerator's generate_proofs_for_rule method.

        Args:
            generator (KGenerator): The KGenerator instance to use.
            rule_names (Optional[List[str]]): List of rule names to visualize.
                                              If None, visualizes all rules.
            max_proofs_per_rule (int): Maximum number of proof trees to
                                       visualize per rule.
            show_invalid_proofs (bool): Whether to visualize proofs that
                                        violate constraints (default: False).
        """
        if rule_names is None:
            rule_names = list(generator.chainer.all_rules.keys())

        print(f"\n{'=' * 80}")
        print(f"VISUALIZING PROOF TREES WITH VARIABLE SUBSTITUTIONS AND CONSTRAINTS")
        print(f"{'=' * 80}")

        for rule_name in rule_names:
            if rule_name not in generator.chainer.all_rules:
                print(f"\nWarning: Rule '{rule_name}' not found. Skipping.")
                continue

            print(f"\n{'-' * 80}")
            print(f"Rule: {rule_name}")
            print(f"{'-' * 80}")

            # REFACTORED: Use KGenerator's generate_proofs_for_rule method
            try:
                proofs = generator.generate_proofs_for_rule(
                    rule_name=rule_name,
                    max_proofs=max_proofs_per_rule,
                )

                if not proofs:
                    print(f"No valid proofs generated for rule: {rule_name}")
                    continue

                print(
                    f"\nGenerated {len(proofs)} valid proof tree(s) for rule: {rule_name}"
                )

                # Visualize each proof
                for i, proof in enumerate(proofs):
                    # Validate this proof to get detailed info
                    all_atoms = generator.chainer._collect_all_atoms(proof)
                    is_valid = generator.chainer._check_constraints(proof, all_atoms)

                    # Skip invalid proofs unless requested
                    if not is_valid and not show_invalid_proofs:
                        continue

                    # Store validation result
                    self.constraint_validation[proof] = is_valid

                    # Determine proof status for filename
                    status = "valid" if is_valid else "invalid"
                    proof_id = f"{rule_name}_proof_{i + 1}_{status}"

                    print(
                        f"\n--- Proof Tree {i + 1}/{len(proofs)} ({status.upper()}) ---"
                    )

                    # Infer substitutions by comparing rule patterns with ground atoms
                    self._infer_substitutions(proof)

                    # Text visualization (to console)
                    self.print_proof_tree_text(proof)

                    # Save text to file
                    text_file = os.path.join(
                        self.output_dir, f"{proof_id}_detailed.txt"
                    )
                    with open(text_file, "w") as f:
                        f.write(f"Proof Tree for Rule: {rule_name}\n")
                        f.write(f"Status: {status.upper()}\n")
                        f.write(f"{'=' * 80}\n\n")

                        # Add constraint validation summary
                        if not is_valid:
                            f.write("CONSTRAINT VIOLATIONS:\n")
                            violations = self.constraint_violations.get(
                                proof, ["Unknown violation"]
                            )
                            for v in violations:
                                f.write(f"  - {v}\n")
                            f.write("\n")

                        f.write(self.get_proof_tree_text(proof))
                    print(f"✓ Saved text visualization to: {text_file}")

                    # Graphviz visualization (if available)
                    self.create_graphviz_visualization(proof, proof_id)

                    # Print statistics
                    stats = self.get_proof_statistics(proof)
                    print(f"\nProof Statistics:")
                    print(f"  Status: {status.upper()}")
                    print(f"  Total nodes: {stats['total_nodes']}")
                    print(f"  Base facts: {stats['base_facts']}")
                    print(f"  Derived facts: {stats['derived_facts']}")
                    print(f"  Max depth: {stats['max_depth']}")
                    print(f"  Unique individuals: {stats['unique_individuals']}")
                    print(f"  Rules used: {', '.join(stats['rules_used'])}")

                    if not is_valid:
                        print(f"\n  ⚠️  CONSTRAINT VIOLATIONS:")
                        violations = self.constraint_violations.get(
                            proof, ["Unknown violation"]
                        )
                        for v in violations:
                            print(f"    - {v}")

            except Exception as e:
                print(f"\nError visualizing proofs for {rule_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    def _infer_substitutions(
        self, proof: Proof, parent_subs: Optional[Dict[Var, str]] = None
    ) -> Dict[Var, str]:
        """
        Recursively infer variable substitutions by comparing rule patterns
        with ground atoms.

        This method reconstructs the variable bindings that the chainer used
        by matching rule patterns (with variables) against ground instances
        (with individuals).

        Args:
            proof (Proof): Current proof node to analyze.
            parent_subs (Optional[Dict[Var, str]]): Substitutions inherited
                                                     from parent proof.

        Returns:
            Dict[Var, str]: Dictionary mapping variables to their ground values.
        """
        if parent_subs is None:
            parent_subs = {}

        # Start with parent substitutions
        current_subs = parent_subs.copy()

        if proof.rule is not None:
            # Match rule conclusion pattern against ground goal
            conclusion_pattern = proof.rule.conclusion
            ground_goal = proof.goal

            # Infer substitutions from unification
            new_subs = self._match_pattern_to_ground(conclusion_pattern, ground_goal)
            current_subs.update(new_subs)

            # Also infer from premises (if they exist)
            if len(proof.sub_proofs) == len(proof.rule.premises):
                for premise_pattern, sub_proof in zip(
                    proof.rule.premises, proof.sub_proofs
                ):
                    premise_ground = sub_proof.goal
                    premise_subs = self._match_pattern_to_ground(
                        premise_pattern, premise_ground
                    )
                    current_subs.update(premise_subs)

        # Store substitutions for this proof node
        self.substitution_history[proof] = current_subs

        # Recurse to sub-proofs
        for sub_proof in proof.sub_proofs:
            self._infer_substitutions(sub_proof, current_subs)

        return current_subs

    def _match_pattern_to_ground(self, pattern: Atom, ground: Atom) -> Dict[Var, str]:
        """
        Match a pattern atom (with variables) against a ground atom.

        Returns mapping from variables to ground terms by aligning the
        pattern with the ground instance.

        Example:
            pattern = Atom(Var('X'), hasParent, Var('Y'))
            ground  = Atom(Ind_0, hasParent, Ind_1)
            result  = {Var('X'): 'Ind_0', Var('Y'): 'Ind_1'}

        Args:
            pattern (Atom): Atom with variables (from rule).
            ground (Atom): Ground atom (actual instance).

        Returns:
            Dict[Var, str]: Dictionary mapping Var to ground term names.
        """
        subs = {}

        # Match subject
        if isinstance(pattern.subject, Var):
            subs[pattern.subject] = self._format_term(ground.subject)

        # Match predicate (usually not a var, but check anyway)
        if isinstance(pattern.predicate, Var):
            subs[pattern.predicate] = self._format_term(ground.predicate)

        # Match object
        if isinstance(pattern.object, Var):
            subs[pattern.object] = self._format_term(ground.object)

        return subs

    def print_proof_tree_text(self, proof: Proof, indent: int = 0) -> None:
        """
        Print proof tree with variable substitutions to console.

        Displays a hierarchical view of the proof tree showing:
        - Goal atoms (what's being proved)
        - Rule applications
        - Variable substitutions
        - Recursion tracking
        - Sub-proofs
        - Constraint validation status

        Args:
            proof (Proof): The proof tree to print.
            indent (int): Current indentation level (for hierarchical display).
        """
        prefix = "  " * indent

        # Get substitutions for this proof
        subs = self.substitution_history.get(proof, {})

        # Get validation status
        is_valid = self.constraint_validation.get(proof, True)
        status_marker = "✓" if is_valid else "✗"

        # Format the goal
        goal_str = self._format_atom(proof.goal)

        if proof.rule is None:
            # Base fact - no rule derivation
            print(f"{prefix}[BASE FACT] {status_marker} {goal_str}")
        else:
            # Derived fact - show full derivation details
            print(f"{prefix}[DERIVE] {status_marker} {goal_str}")
            print(f"{prefix}  via rule: {proof.rule.name}")

            # Show the original rule pattern
            rule_pattern = f"{self._format_atom(proof.rule.conclusion)}"
            print(f"{prefix}  pattern: {rule_pattern}")

            # Show variable substitutions
            if subs:
                print(f"{prefix}  substitutions:")
                for var, value in sorted(subs.items(), key=lambda x: x[0].name):
                    print(f"{prefix}    {var.name} → {value}")

            # Show recursion tracking info (if applicable)
            if proof.recursive_use_counts:
                counts_str = ", ".join(
                    [f"{name}:{count}" for name, count in proof.recursive_use_counts]
                )
                print(f"{prefix}  recursion: {counts_str}")

            # Show premises and their proofs
            if proof.sub_proofs:
                print(f"{prefix}  from premises:")
                for i, (premise_pattern, sub_proof) in enumerate(
                    zip(proof.rule.premises, proof.sub_proofs)
                ):
                    premise_pattern_str = self._format_atom(premise_pattern)
                    premise_ground_str = self._format_atom(sub_proof.goal)
                    print(f"{prefix}    [{i + 1}] pattern: {premise_pattern_str}")
                    print(f"{prefix}        ground:  {premise_ground_str}")
                    # Recursively print sub-proof
                    self.print_proof_tree_text(sub_proof, indent + 3)

    def get_proof_tree_text(self, proof: Proof, indent: int = 0) -> str:
        """
        Get proof tree with substitutions as formatted text string.

        This is similar to print_proof_tree_text but returns a string
        instead of printing, useful for saving to files.

        Args:
            proof (Proof): The proof tree.
            indent (int): Current indentation level.

        Returns:
            str: Formatted string representation of the proof tree.
        """
        lines = []
        prefix = "  " * indent

        # Get substitutions
        subs = self.substitution_history.get(proof, {})

        # Get validation status
        is_valid = self.constraint_validation.get(proof, True)
        status_marker = "✓" if is_valid else "✗"

        # Format the goal
        goal_str = self._format_atom(proof.goal)

        if proof.rule is None:
            lines.append(f"{prefix}[BASE FACT] {status_marker} {goal_str}")
        else:
            lines.append(f"{prefix}[DERIVE] {status_marker} {goal_str}")
            lines.append(f"{prefix}  via rule: {proof.rule.name}")

            # Show original rule pattern
            rule_pattern = f"{self._format_atom(proof.rule.conclusion)}"
            lines.append(f"{prefix}  pattern: {rule_pattern}")

            # Show variable substitutions
            if subs:
                lines.append(f"{prefix}  substitutions:")
                for var, value in sorted(subs.items(), key=lambda x: x[0].name):
                    lines.append(f"{prefix}    {var.name} → {value}")

            # Show recursion info if applicable
            if proof.recursive_use_counts:
                counts_str = ", ".join(
                    [f"{name}:{count}" for name, count in proof.recursive_use_counts]
                )
                lines.append(f"{prefix}  recursion: {counts_str}")

            # Show premises
            if proof.sub_proofs:
                lines.append(f"{prefix}  from premises:")
                for i, (premise_pattern, sub_proof) in enumerate(
                    zip(proof.rule.premises, proof.sub_proofs)
                ):
                    premise_pattern_str = self._format_atom(premise_pattern)
                    premise_ground_str = self._format_atom(sub_proof.goal)
                    lines.append(
                        f"{prefix}    [{i + 1}] pattern: {premise_pattern_str}"
                    )
                    lines.append(f"{prefix}        ground:  {premise_ground_str}")
                    # Recursively get sub-proof text
                    lines.append(self.get_proof_tree_text(sub_proof, indent + 3))

        return "\n".join(lines)

    def create_graphviz_visualization(self, proof: Proof, proof_id: str) -> None:
        """
        Create a Graphviz visualization with variable substitutions and constraint status.

        Generates a directed graph showing the proof tree structure with:
        - Color-coded nodes (green for base facts, blue for derived, red for constraint violations)
        - Variable substitutions displayed in each node
        - Premise patterns shown on edges
        - Recursion tracking information
        - Constraint validation status

        Args:
            proof (Proof): The proof tree to visualize.
            proof_id (str): Identifier for this proof (used in filename).
        """
        # Reset node tracking for new graph
        self.node_counter = 0
        self.node_ids = {}

        # Get validation status for title
        is_valid = self.constraint_validation.get(proof, True)
        status = "VALID" if is_valid else "INVALID (Constraint Violation)"

        # Create graph with sensible defaults
        dot = graphviz.Digraph(comment=f"Proof Tree: {proof_id}")
        dot.attr(rankdir="TB")  # Top to bottom layout
        dot.attr("node", shape="box", style="rounded,filled")
        dot.attr("graph", fontname="Courier", fontsize="10")
        dot.attr("node", fontname="Courier", fontsize="9")
        dot.attr("edge", fontname="Courier", fontsize="8")

        # Add title with validation status
        dot.attr(label=f"Proof Tree: {proof_id}\\nStatus: {status}", labelloc="t")

        # Build the graph recursively
        self._add_proof_to_graph(proof, dot, None)

        # Save to file
        output_path = os.path.join(self.output_dir, proof_id + "_detailed")
        try:
            dot.render(output_path, format="pdf", cleanup=True)
            print(f"✓ Saved detailed graph to: {output_path}.pdf")
        except Exception as e:
            print(f"✗ Failed to create graph: {e}")
            # Save .dot file as fallback
            dot.save(output_path + ".dot")
            print(f"  Saved .dot file to: {output_path}.dot")

    def _add_proof_to_graph(
        self, proof: Proof, dot: graphviz.Digraph, parent_id: Optional[str]
    ) -> str:
        """
        Recursively add proof nodes with substitution info to graphviz graph.

        Creates a node for this proof and recursively adds nodes for all
        sub-proofs, connecting them with labeled edges. Nodes are color-coded
        based on constraint validation status.

        Args:
            proof (Proof): Current proof node to add.
            dot (graphviz.Digraph): Graphviz graph object to add nodes to.
            parent_id (Optional[str]): ID of parent node (None for root).

        Returns:
            str: Node ID for this proof (for parent to reference).
        """
        # Reuse node if already created (DAG structure)
        if proof in self.node_ids:
            return self.node_ids[proof]

        # Create unique node ID
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        self.node_ids[proof] = node_id

        # Get substitutions for this proof
        subs = self.substitution_history.get(proof, {})

        # Get validation status
        is_valid = self.constraint_validation.get(proof, True)

        # Format node label
        goal_str = self._format_atom(proof.goal)

        if proof.rule is None:
            # Base fact - simple green box (or red if invalid)
            color = "lightgreen" if is_valid else "lightcoral"
            label = f"BASE FACT\\n{goal_str}"
            if not is_valid:
                label += "\\n✗ INVALID"
            dot.node(node_id, label, fillcolor=color)
        else:
            # Derived fact - blue box with detailed info (or red if invalid)
            color = "lightblue" if is_valid else "lightcoral"
            # Use \\l for left-aligned text in graphviz
            # Use \\n for new lines
            label = f"DERIVE\\n{goal_str}\\n\\l"
            if not is_valid:
                label = f"DERIVE (✗ INVALID)\\n{goal_str}\\n\\l"
            label += f"Rule: {proof.rule.name}\\l"
            label += f"Pattern: {self._format_atom(proof.rule.conclusion)}\\l"

            # Add substitutions to label
            if subs:
                label += "\\lSubstitutions:\\l"
                for var, value in sorted(subs.items(), key=lambda x: x[0].name):
                    label += f"  {var.name} → {value}\\l"

            # Add recursion info if present
            if proof.recursive_use_counts:
                rec_info = ", ".join(
                    [f"{name}:{count}" for name, count in proof.recursive_use_counts]
                )
                label += f"\\lRecursion: {rec_info}\\l"

            dot.node(node_id, label, fillcolor=color)

            # Add edges to sub-proofs with premise patterns as labels
            for i, (premise_pattern, sub_proof) in enumerate(
                zip(proof.rule.premises, proof.sub_proofs)
            ):
                sub_id = self._add_proof_to_graph(sub_proof, dot, node_id)

                # Edge label shows which premise this sub-proof satisfies
                edge_label = f"premise {i + 1}:\\n{self._format_atom(premise_pattern)}"
                dot.edge(node_id, sub_id, label=edge_label)

        return node_id

    def _format_atom(self, atom: Atom) -> str:
        """
        Format an atom for display in a human-readable way.

        Args:
            atom (Atom): The atom to format.

        Returns:
            str: Formatted string like "(Ind_0, hasParent, Ind_1)".
        """
        s = self._format_term(atom.subject)
        p = self._format_term(atom.predicate)
        o = self._format_term(atom.object)
        return f"({s}, {p}, {o})"

    def _format_term(self, term) -> str:
        """
        Format a term (Individual, Class, Relation, Var, etc.) for display.

        Args:
            term: The term to format.

        Returns:
            str: Human-readable string representation.
        """
        if isinstance(term, (Individual, Class, Relation)):
            return term.name
        elif isinstance(term, Var):
            return f"{term.name}"
        elif term == RDF.type:
            return "rdf:type"
        else:
            return str(term)

    def get_proof_statistics(self, proof: Proof) -> Dict:
        """
        Calculate statistics about a proof tree.

        Traverses the entire proof tree and collects:
        - Node counts (total, base, derived)
        - Depth information
        - Individual tracking
        - Rule usage
        - Variable information
        - Constraint validation status

        Args:
            proof (Proof): The proof tree to analyze.

        Returns:
            Dict: Dictionary of statistics.
        """
        stats = {
            "total_nodes": 0,
            "base_facts": 0,
            "derived_facts": 0,
            "max_depth": 0,
            "unique_individuals": set(),
            "rules_used": set(),
            "total_substitutions": 0,
            "unique_variables": set(),
            "is_valid": self.constraint_validation.get(proof, True),
        }

        def traverse(p: Proof, depth: int):
            """Recursively traverse proof tree and collect stats."""
            stats["total_nodes"] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)

            if p.rule is None:
                stats["base_facts"] += 1
            else:
                stats["derived_facts"] += 1
                stats["rules_used"].add(p.rule.name)

            # Collect individuals from goal
            for term in [p.goal.subject, p.goal.object]:
                if isinstance(term, Individual):
                    stats["unique_individuals"].add(term.name)

            # Collect substitution info
            subs = self.substitution_history.get(p, {})
            stats["total_substitutions"] += len(subs)
            for var in subs.keys():
                stats["unique_variables"].add(var.name)

            # Recurse to sub-proofs
            for sub in p.sub_proofs:
                traverse(sub, depth + 1)

        # Start traversal from root
        traverse(proof, 0)

        # Convert sets to counts and sorted lists
        stats["unique_individuals"] = len(stats["unique_individuals"])
        stats["unique_variables"] = len(stats["unique_variables"])
        stats["rules_used"] = sorted(list(stats["rules_used"]))

        return stats


def main():
    """
    Main entry point for the visualizer.

    Parses command-line arguments, loads ontology, runs backward chainer,
    and generates visualizations for proof trees with constraint validation.
    """
    # ==================== PARSE ARGUMENTS ==================== #

    parser = argparse.ArgumentParser(
        description="Visualize proof trees with variable substitutions and constraints"
    )
    parser.add_argument(
        "--ontology-path",
        type=str,
        default="data/toy.ttl",
        help="Path to the ontology file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--max-recursion", type=int, default=5, help="Maximum recursion depth"
    )
    parser.add_argument(
        "--max-proofs",
        type=int,
        default=10,
        help="Maximum number of proofs to visualize per rule",
    )
    parser.add_argument(
        "--rules",
        type=str,
        nargs="+",
        default=None,
        help="Specific rule names to visualize (default: all rules)",
    )
    parser.add_argument(
        "--show-invalid",
        action="store_true",
        help="Show proofs that violate constraints (default: only valid proofs)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # ==================== RUN VISUALIZATION PIPELINE ==================== #

    try:
        # REFACTORED: Use KGenerator instead of creating parser/chainer directly
        print(f"Initializing KGenerator from: {args.ontology_path}")
        generator = KGenerator(
            ontology_file=args.ontology_path,
            max_recursion=args.max_recursion,
            verbose=args.verbose,
        )

        print(f"Loaded {len(generator.parser.constraints)} constraints")

        # Create visualizer
        visualizer = ProofTreeVisualizer(output_dir=args.output_dir)

        # Visualize proofs
        visualizer.visualize_all_proofs(
            generator=generator,
            rule_names=args.rules,
            max_proofs_per_rule=args.max_proofs,
            show_invalid_proofs=args.show_invalid,
        )

        print(f"\n{'=' * 80}")
        print(f"VISUALIZATION COMPLETE")
        print(f"Output saved to: {args.output_dir}")
        print(f"{'=' * 80}\n")

    # ==================== ERROR HANDLING ==================== #

    except FileNotFoundError:
        print(
            f"Error: Ontology file not found at '{args.ontology_path}'",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred:\n{e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
