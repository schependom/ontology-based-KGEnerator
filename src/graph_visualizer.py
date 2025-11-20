"""
DESCRIPTION:

    Visualizes the Knowledge Graph using Graphviz.

AUTHOR:

    Vincent Van Schependom
"""

import os
from collections import defaultdict
import graphviz

from data_structures import KnowledgeGraph


class GraphVisualizer:
    """
    Visualizes the Knowledge Graph using Graphviz.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize(self, kg: KnowledgeGraph, filename: str):
        """
        Exports the Knowledge Graph as an image using Graphviz.
        """
        # Remove extension as graphviz adds it based on format
        name = os.path.splitext(filename)[0]
        file_format = os.path.splitext(filename)[1][1:] or "png"

        dot = graphviz.Digraph(name=name, format=file_format)

        # Use 'neato' (spring model) or 'fdp' for better non-hierarchical layout
        # 'overlap=false' helps prevent node overlaps
        # 'splines=true' makes edges curved to avoid passing through nodes
        dot.engine = "neato"
        dot.attr(overlap="false")
        dot.attr(splines="true")
        dot.attr(sep="+25")  # Minimum distance between nodes

        # Node attributes
        dot.attr(
            "node",
            shape="circle",
            style="filled",
            fillcolor="lightblue",
            fontname="Helvetica",
            fontsize="12",
            margin="0.1",
        )

        # Edge attributes
        dot.attr(
            "edge", fontname="Helvetica", fontsize="10", len="2.0"
        )  # Target edge length for neato

        # Collect memberships per individual
        individual_classes = defaultdict(list)
        for membership in kg.memberships:
            if membership.is_member:
                individual_classes[membership.individual.name].append(
                    membership.cls.name
                )

        # Add nodes
        for ind in kg.individuals:
            label = ind.name
            classes = individual_classes.get(ind.name, [])
            if classes:
                label += f"\n({', '.join(classes)})"

            dot.node(ind.name, label=label)

        # Group edges by (source, target) to merge labels
        edges = defaultdict(lambda: {"positive": [], "negative": []})

        for triple in kg.triples:
            key = (triple.subject.name, triple.object.name)
            if triple.positive:
                edges[key]["positive"].append(triple.predicate.name)
            else:
                edges[key]["negative"].append(triple.predicate.name)

        # Add edges
        for (u, v), data in edges.items():
            # Add positive edges (merged)
            if data["positive"]:
                # Deduplicate labels
                labels = sorted(list(set(data["positive"])))
                label = "\n".join(labels)
                dot.edge(u, v, label=label, color="black")

            # Add negative edges (merged)
            if data["negative"]:
                # Deduplicate labels
                labels = sorted(list(set(data["negative"])))
                label = "\n".join(labels)
                dot.edge(
                    u, v, label=label, color="red", style="dashed", fontcolor="red"
                )

        output_path = os.path.join(self.output_dir, name)
        try:
            # render() saves the source file and then renders it
            output_file = dot.render(output_path, cleanup=True)
            print(f"Graph saved to {output_file}")
        except Exception as e:
            print(f"Error rendering graph with Graphviz: {e}")
            print(
                "Ensure Graphviz is installed on your system (e.g., 'brew install graphviz')."
            )
