"""
DESCRIPTION:

    Enhanced Proof Tree Visualizer with Variable Substitutions.

    Shows:
    - Complete proof tree structure
    - Variable substitutions at each step
    - Individual generation tracking
    - Rule applications with detailed variable bindings

USAGE:

    python visualizer_var.py --ontology-path data/toy.ttl --output-dir output/

AUTHOR:

    Vincent Van Schependom
"""

import argparse
import os
import sys
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

# For graph visualization
try:
    import graphviz

    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Warning: graphviz not installed. Install with: pip install graphviz")

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
from parser import OntologyParser
from verbose_chainer import BackwardChainer
from rdflib.namespace import RDF


class ProofTreeVisualizerV2:
    """
    Enhanced visualizer that tracks and displays variable substitutions
    throughout the proof tree.
    """

    def __init__(self, output_dir: str = "output"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Track nodes for graphviz
        self.node_counter = 0
        self.node_ids: Dict[Proof, str] = {}

        # Track variable substitutions throughout the proof
        self.substitution_history: Dict[Proof, Dict[Var, str]] = {}

    def visualize_all_proofs(
        self,
        chainer: BackwardChainer,
        rule_names: Optional[List[str]] = None,
        max_proofs_per_rule: int = 5,
    ) -> None:
        """
        Generate visualizations for all rules (or specified rules).

        Args:
            chainer: The BackwardChainer instance
            rule_names: List of rule names to visualize (None = all rules)
            max_proofs_per_rule: Maximum number of proof trees to visualize per rule
        """
        if rule_names is None:
            rule_names = list(chainer.all_rules.keys())

        print(f"\n{'=' * 80}")
        print(f"VISUALIZING PROOF TREES WITH VARIABLE SUBSTITUTIONS")
        print(f"{'=' * 80}")

        for rule_name in rule_names:
            if rule_name not in chainer.all_rules:
                print(f"\nWarning: Rule '{rule_name}' not found. Skipping.")
                continue

            print(f"\n{'-' * 80}")
            print(f"Rule: {rule_name}")
            print(f"{'-' * 80}")

            # Generate proofs
            proof_generator = chainer.generate_proof_trees(rule_name)
            proofs = []

            try:
                for i, proof in enumerate(proof_generator):
                    if i >= max_proofs_per_rule:
                        print(
                            f"\nReached max proofs limit ({max_proofs_per_rule}). Stopping generation."
                        )
                        break
                    proofs.append(proof)

            except Exception as e:
                print(f"\nError generating proofs for {rule_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

            if not proofs:
                print(f"No proofs generated for rule: {rule_name}")
                continue

            print(f"\nGenerated {len(proofs)} proof tree(s) for rule: {rule_name}")

            # Visualize each proof
            for i, proof in enumerate(proofs):
                proof_id = f"{rule_name}_proof_{i + 1}"
                print(f"\n--- Proof Tree {i + 1}/{len(proofs)} ---")

                # Infer substitutions by comparing rule patterns with ground atoms
                self._infer_substitutions(proof)

                # Text visualization
                self.print_proof_tree_text(proof)

                # Save text to file
                text_file = os.path.join(self.output_dir, f"{proof_id}_detailed.txt")
                with open(text_file, "w") as f:
                    f.write(f"Proof Tree for Rule: {rule_name}\n")
                    f.write(f"{'=' * 80}\n\n")
                    f.write(self.get_proof_tree_text(proof))
                print(f"✓ Saved text visualization to: {text_file}")

                # Graphviz visualization
                if HAS_GRAPHVIZ:
                    self.create_graphviz_visualization(proof, proof_id)

                # Statistics
                stats = self.get_proof_statistics(proof)
                print(f"\nProof Statistics:")
                print(f"  Total nodes: {stats['total_nodes']}")
                print(f"  Base facts: {stats['base_facts']}")
                print(f"  Derived facts: {stats['derived_facts']}")
                print(f"  Max depth: {stats['max_depth']}")
                print(f"  Unique individuals: {stats['unique_individuals']}")
                print(f"  Rules used: {', '.join(stats['rules_used'])}")

    def _infer_substitutions(
        self, proof: Proof, parent_subs: Optional[Dict[Var, str]] = None
    ) -> Dict[Var, str]:
        """
        Recursively infer variable substitutions by comparing rule patterns with ground atoms.

        Args:
            proof: Current proof node
            parent_subs: Substitutions from parent (inherited)

        Returns:
            Dictionary mapping variables to their ground values
        """
        if parent_subs is None:
            parent_subs = {}

        current_subs = parent_subs.copy()

        if proof.rule is not None:
            # Match rule conclusion pattern against ground goal
            conclusion_pattern = proof.rule.conclusion
            ground_goal = proof.goal

            # Infer substitutions from the unification
            new_subs = self._match_pattern_to_ground(conclusion_pattern, ground_goal)
            current_subs.update(new_subs)

            # Also infer from premises
            if len(proof.sub_proofs) == len(proof.rule.premises):
                for premise_pattern, sub_proof in zip(
                    proof.rule.premises, proof.sub_proofs
                ):
                    premise_ground = sub_proof.goal
                    premise_subs = self._match_pattern_to_ground(
                        premise_pattern, premise_ground
                    )
                    current_subs.update(premise_subs)

        # Store substitutions for this proof
        self.substitution_history[proof] = current_subs

        # Recurse to sub-proofs
        for sub_proof in proof.sub_proofs:
            self._infer_substitutions(sub_proof, current_subs)

        return current_subs

    def _match_pattern_to_ground(self, pattern: Atom, ground: Atom) -> Dict[Var, str]:
        """
        Match a pattern atom (with variables) against a ground atom.
        Returns mapping from variables to ground terms.

        Args:
            pattern: Atom with variables (from rule)
            ground: Ground atom (actual instance)

        Returns:
            Dictionary mapping Var to ground term names
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

        Args:
            proof: The proof tree to print
            indent: Current indentation level
        """
        prefix = "  " * indent

        # Get substitutions for this proof
        subs = self.substitution_history.get(proof, {})

        # Print the goal
        goal_str = self._format_atom(proof.goal)

        if proof.rule is None:
            print(f"{prefix}[BASE FACT] {goal_str}")
        else:
            print(f"{prefix}[DERIVE] {goal_str}")
            print(f"{prefix}  via rule: {proof.rule.name}")

            # Show the original rule pattern
            rule_pattern = f"{self._format_atom(proof.rule.conclusion)}"
            print(f"{prefix}  pattern: {rule_pattern}")

            # Show variable substitutions
            if subs:
                print(f"{prefix}  substitutions:")
                for var, value in sorted(subs.items(), key=lambda x: x[0].name):
                    print(f"{prefix}    {var.name} → {value}")

            # Show recursion info
            if proof.recursive_use_counts:
                counts_str = ", ".join(
                    [f"{name}:{count}" for name, count in proof.recursive_use_counts]
                )
                print(f"{prefix}  recursion: {counts_str}")

            # Show premises
            if proof.sub_proofs:
                print(f"{prefix}  from premises:")
                for i, (premise_pattern, sub_proof) in enumerate(
                    zip(proof.rule.premises, proof.sub_proofs)
                ):
                    premise_pattern_str = self._format_atom(premise_pattern)
                    premise_ground_str = self._format_atom(sub_proof.goal)
                    print(f"{prefix}    [{i + 1}] pattern: {premise_pattern_str}")
                    print(f"{prefix}        ground:  {premise_ground_str}")
                    self.print_proof_tree_text(sub_proof, indent + 3)

    def get_proof_tree_text(self, proof: Proof, indent: int = 0) -> str:
        """
        Get proof tree with substitutions as formatted text string.

        Args:
            proof: The proof tree
            indent: Current indentation level

        Returns:
            Formatted string representation
        """
        lines = []
        prefix = "  " * indent

        # Get substitutions
        subs = self.substitution_history.get(proof, {})

        # Format the goal
        goal_str = self._format_atom(proof.goal)

        if proof.rule is None:
            lines.append(f"{prefix}[BASE FACT] {goal_str}")
        else:
            lines.append(f"{prefix}[DERIVE] {goal_str}")
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
                    lines.append(self.get_proof_tree_text(sub_proof, indent + 3))

        return "\n".join(lines)

    def create_graphviz_visualization(self, proof: Proof, proof_id: str) -> None:
        """
        Create a Graphviz visualization with variable substitutions.

        Args:
            proof: The proof tree
            proof_id: Identifier for this proof (used in filename)
        """
        # Reset node tracking
        self.node_counter = 0
        self.node_ids = {}

        # Create graph
        dot = graphviz.Digraph(comment=f"Proof Tree: {proof_id}")
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box", style="rounded,filled")
        dot.attr("graph", fontname="Courier", fontsize="10")
        dot.attr("node", fontname="Courier", fontsize="9")
        dot.attr("edge", fontname="Courier", fontsize="8")

        # Build the graph
        self._add_proof_to_graph_v2(proof, dot, None)

        # Save
        output_path = os.path.join(self.output_dir, proof_id + "_detailed")
        try:
            dot.render(output_path, format="pdf", cleanup=True)
            print(f"✓ Saved detailed graph to: {output_path}.pdf")
        except Exception as e:
            print(f"✗ Failed to create graph: {e}")
            dot.save(output_path + ".dot")
            print(f"  Saved .dot file to: {output_path}.dot")

    def _add_proof_to_graph_v2(
        self, proof: Proof, dot: graphviz.Digraph, parent_id: Optional[str]
    ) -> str:
        """
        Recursively add proof nodes with substitution info to graphviz graph.

        Args:
            proof: Current proof node
            dot: Graphviz graph object
            parent_id: ID of parent node (None for root)

        Returns:
            Node ID for this proof
        """
        # Get or create node ID
        if proof in self.node_ids:
            return self.node_ids[proof]

        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        self.node_ids[proof] = node_id

        # Get substitutions
        subs = self.substitution_history.get(proof, {})

        # Format node label
        goal_str = self._format_atom(proof.goal)

        if proof.rule is None:
            # Base fact - green box
            label = f"BASE FACT\\n{goal_str}"
            dot.node(node_id, label, fillcolor="lightgreen")
        else:
            # Derived fact - blue box with substitutions
            label = f"DERIVE\\n{goal_str}\\l\\l"  # \\l = left-aligned
            label += f"Rule: {proof.rule.name}\\l"
            label += f"Pattern: {self._format_atom(proof.rule.conclusion)}\\l"

            # Add substitutions
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

            dot.node(node_id, label, fillcolor="lightblue")

            # Add edges to sub-proofs with premise patterns
            for i, (premise_pattern, sub_proof) in enumerate(
                zip(proof.rule.premises, proof.sub_proofs)
            ):
                sub_id = self._add_proof_to_graph_v2(sub_proof, dot, node_id)

                # Edge label shows premise pattern
                edge_label = f"premise {i + 1}:\\n{self._format_atom(premise_pattern)}"
                dot.edge(node_id, sub_id, label=edge_label)

        return node_id

    def _format_atom(self, atom: Atom) -> str:
        """Format an atom for display."""
        s = self._format_term(atom.subject)
        p = self._format_term(atom.predicate)
        o = self._format_term(atom.object)
        return f"({s}, {p}, {o})"

    def _format_term(self, term) -> str:
        """Format a term for display."""
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

        Args:
            proof: The proof tree

        Returns:
            Dictionary of statistics
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
        }

        def traverse(p: Proof, depth: int):
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

            # Recurse
            for sub in p.sub_proofs:
                traverse(sub, depth + 1)

        traverse(proof, 0)
        stats["unique_individuals"] = len(stats["unique_individuals"])
        stats["unique_variables"] = len(stats["unique_variables"])
        stats["rules_used"] = sorted(list(stats["rules_used"]))

        return stats


def main():
    """Main entry point."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Visualize proof trees with variable substitutions"
    )
    parser.add_argument(
        "--ontology-path",
        type=str,
        default="data/family.ttl",
        help="Path to the ontology file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--max-recursion", type=int, default=2, help="Maximum recursion depth"
    )
    parser.add_argument(
        "--max-proofs",
        type=int,
        default=5,
        help="Maximum number of proofs to visualize per rule",
    )
    parser.add_argument(
        "--rules",
        type=str,
        nargs="+",
        default=None,
        help="Specific rule names to visualize (default: all rules)",
    )

    args = parser.parse_args()

    # Check graphviz
    if not HAS_GRAPHVIZ:
        print("\n" + "!" * 80)
        print("WARNING: graphviz not installed. Only text output will be available.")
        print("Install with: pip install graphviz")
        print("You also need graphviz system package: https://graphviz.org/download/")
        print("!" * 80 + "\n")

    try:
        # Parse ontology
        print(f"Loading ontology from: {args.ontology_path}")
        onto_parser = OntologyParser(args.ontology_path)

        # Initialize chainer
        print(f"Initializing backward chainer (max recursion: {args.max_recursion})")
        chainer = BackwardChainer(
            all_rules=onto_parser.rules, max_recursion_depth=args.max_recursion
        )

        # Create visualizer
        visualizer = ProofTreeVisualizerV2(output_dir=args.output_dir)

        # Visualize proofs
        visualizer.visualize_all_proofs(
            chainer=chainer, rule_names=args.rules, max_proofs_per_rule=args.max_proofs
        )

        print(f"\n{'=' * 80}")
        print(f"VISUALIZATION COMPLETE")
        print(f"Output saved to: {args.output_dir}")
        print(f"{'=' * 80}\n")

    except FileNotFoundError:
        print(
            f"Error: Ontology file not found at '{args.ontology_path}'", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred:\n{e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
