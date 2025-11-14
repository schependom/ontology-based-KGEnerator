"""
DESCRIPTION:

    Main script to generate facts using backward chaining on a given ontology.

AUTHOR

    Vincent Van Schependom
"""

import sys
import argparse
import traceback
from typing import Dict, Set
from rdflib.namespace import RDF

from data_structures import (
    KnowledgeGraph,
    Individual,
    Class,
    Relation,
    Attribute,
    Triple,
    Membership,
    AttributeTriple,
    Atom,
    Proof,
    Term,
)
from parser import OntologyParser
from chainer import BackwardChainer


class KGenerator:
    """
    Knowledge Graph data Generator (KGenerator).
    Orchestrates the full data generation pipeline.
    """

    def __init__(
        self, ontology_file: str, max_recursion: int = 2, max_proofs_per_rule: int = 100
    ):
        """
        Initializes the parser and chainer.

        Args:
            ontology_file (str): Path to the .ttl ontology file.
            max_recursion (int): The maximum depth for recursive rules.
            max_proofs_per_rule (int): Maximum number of proofs per starting rule.
        """

        print(f"Loading and parsing ontology from: {ontology_file}")

        self.parser = OntologyParser(ontology_file)
        self.chainer = BackwardChainer(
            all_rules=self.parser.rules,
            max_recursion_depth=max_recursion,
            max_proofs_per_rule=max_proofs_per_rule,
        )

        # Store schemas from the parser
        self.schema_classes = self.parser.classes
        self.schema_relations = self.parser.relations
        self.schema_attributes = self.parser.attributes

        # Store the data
        self.individuals: Dict[str, Individual] = {}
        self.triples: Dict[tuple, Triple] = {}
        self.memberships: Dict[tuple, Membership] = {}
        self.attr_triples: Dict[tuple, AttributeTriple] = {}

        # Track processed proofs to avoid duplicate work
        self.processed_proofs: Set[Proof] = set()

        self.visualized_one_proof = False

    def _register_individual(self, term: Term) -> None:
        """Adds an Individual to our store if it's not present."""
        if isinstance(term, Individual):
            if term.name not in self.individuals:
                self.individuals[term.name] = term
        else:
            print(f"Warning: Expected Individual term, got: {term}")

    def _add_atom_and_proof(self, atom: Atom, proof: Proof) -> None:
        """
        Converts a ground Atom into a KG fact and adds it to our stores.
        If the fact already exists, it appends the proof to the fact's proof list.
        """

        if not atom.is_ground():
            print(f"Warning: Skipping non-ground atom: {atom}")
            return

        s, p, o = atom.subject, atom.predicate, atom.object

        # CLASS MEMBERSHIPS
        if p == RDF.type and isinstance(o, Class):
            self._register_individual(s)
            key = (s.name, o.name)
            if key not in self.memberships:
                self.memberships[key] = Membership(
                    individual=s, cls=o, is_member=True, proofs=[]
                )
            self.memberships[key].proofs.append(proof)

        # RELATIONAL TRIPLES
        elif isinstance(o, Individual) and isinstance(p, Relation):
            self._register_individual(s)
            self._register_individual(o)
            key = (s.name, p.name, o.name)
            if key not in self.triples:
                self.triples[key] = Triple(
                    subject=s, predicate=p, object=o, positive=True, proofs=[]
                )
            self.triples[key].proofs.append(proof)

        # ATTRIBUTE TRIPLES
        elif isinstance(p, Attribute):
            self._register_individual(s)
            key = (s.name, p.name, o)
            if key not in self.attr_triples:
                self.attr_triples[key] = AttributeTriple(
                    subject=s, predicate=p, value=o, proofs=[]
                )
            self.attr_triples[key].proofs.append(proof)

    def generate_data(self) -> KnowledgeGraph:
        """
        Runs the full generation process by iterating through all rules,
        generating all possible proofs, and building the KnowledgeGraph.
        """

        all_rules = self.parser.rules

        if not all_rules:
            print("Warning: No rules found in the ontology. No data will be generated.")
            return self.build_knowledge_graph()

        print(f"Starting data generation from {len(all_rules)} rules...")

        total_top_level_proofs = 0

        for i, rule in enumerate(all_rules):
            print(f"\n[{i + 1}/{len(all_rules)}] Generating from rule: {rule.name}")

            proof_generator = self.chainer.generate_proof_trees(rule.name)

            top_level_proofs = 0

            for top_level_proof in proof_generator:
                top_level_proofs += 1

                # Traverse the entire proof tree
                stack = [top_level_proof]

                while stack:
                    current_proof = stack.pop()

                    if current_proof in self.processed_proofs:
                        continue

                    self.processed_proofs.add(current_proof)

                    # Add the goal of this proof to our KG
                    self._add_atom_and_proof(current_proof.goal, current_proof)

                    # Add all sub-proofs to the stack
                    for sub_proof in current_proof.sub_proofs:
                        stack.append(sub_proof)

            if top_level_proofs == 0:
                print("  -> No proofs found (rule may not be a valid starting point).")
            else:
                print(f"  -> Found and processed {top_level_proofs} proof tree(s).")
                total_top_level_proofs += top_level_proofs

        print(
            f"\n--- Data generation complete. Total proof trees: {total_top_level_proofs} ---"
        )

        return self.build_knowledge_graph()

    def build_knowledge_graph(self) -> KnowledgeGraph:
        """Assembles the final KnowledgeGraph object from all collected data."""
        return KnowledgeGraph(
            attributes=list(self.schema_attributes.values()),
            classes=list(self.schema_classes.values()),
            relations=list(self.schema_relations.values()),
            individuals=list(self.individuals.values()),
            triples=list(self.triples.values()),
            memberships=list(self.memberships.values()),
            attribute_triples=list(self.attr_triples.values()),
        )


def main():
    """Main entry point for the script."""

    default_ontology_path = "data/family.ttl"
    default_max_recursion = 3
    default_max_proofs = 200

    parser = argparse.ArgumentParser(
        description="Ontology-based Knowledge Graph Data Generator"
    )
    parser.add_argument(
        "--ontology-path",
        type=str,
        default=default_ontology_path,
        help=f"Path to the ontology file (default: '{default_ontology_path}')",
    )
    parser.add_argument(
        "--max-recursion",
        type=int,
        default=default_max_recursion,
        help=f"Maximum depth for recursive rules (default: {default_max_recursion})",
    )
    parser.add_argument(
        "--max-proofs",
        type=int,
        default=default_max_proofs,
        help=f"Maximum number of proofs per rule (default: {default_max_proofs})",
    )
    args = parser.parse_args()

    try:
        generator = KGenerator(
            args.ontology_path,
            max_recursion=args.max_recursion,
            max_proofs_per_rule=args.max_proofs,
        )

        kg = generator.generate_data()

        print("\n" + "=" * 60)
        print("KNOWLEDGE GRAPH GENERATION SUMMARY")
        print("=" * 60)
        print(f"\nSchema Elements:")
        print(f"  Classes:    {len(kg.classes)}")
        print(f"  Relations:  {len(kg.relations)}")
        print(f"  Attributes: {len(kg.attributes)}")
        print(f"\nGenerated Data:")
        print(f"  Individuals: {len(kg.individuals)}")
        print(f"  Triples:     {len(kg.triples)}")
        print(f"  Memberships: {len(kg.memberships)}")
        print(f"  AttrTriples: {len(kg.attribute_triples)}")

        # Calculate statistics
        base_triples = sum(1 for t in kg.triples if t.is_base_fact)
        inferred_triples = len(kg.triples) - base_triples
        base_memberships = sum(1 for m in kg.memberships if m.is_base_fact)
        inferred_memberships = len(kg.memberships) - base_memberships

        print(f"\nFact Types:")
        print(f"  Base Facts:     {base_triples + base_memberships}")
        print(f"  Inferred Facts: {inferred_triples + inferred_memberships}")
        print("=" * 60 + "\n")

        kg.print()

    except FileNotFoundError:
        print(
            f"Error: Ontology file not found at '{args.ontology_path}'", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred:\n{e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
