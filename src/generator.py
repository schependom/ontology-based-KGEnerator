import sys
import argparse
from typing import Dict, Set
from rdflib.namespace import RDF

# Import all our custom modules
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


class KnowledgeGraphGenerator:
    """
    Orchestrates the full data generation pipeline:
    1. Parses an ontology.
    2. Uses a BackwardChainer to generate all possible proof trees.
    3. Populates a KnowledgeGraph with the facts from these proofs.
    """

    def __init__(self, ontology_file: str, max_recursion: int = 2):
        """
        Initializes the parser and chainer.

        Args:
            ontology_file (str): Path to the .ttl ontology file.
            max_recursion (int): The maximum depth for recursive rules.
        """
        print(f"Loading and parsing ontology from: {ontology_file}")
        self.parser = OntologyParser(ontology_file)

        self.chainer = BackwardChainer(
            all_rules=self.parser.rules, max_recursion_depth=max_recursion
        )

        # Schema stores from the parser
        self.schema_classes = self.parser.classes
        self.schema_relations = self.parser.relations
        self.schema_attributes = self.parser.attributes

        # Stores for the generated Knowledge Graph data
        # We use dictionaries for deduplication
        self.individuals: Dict[str, Individual] = {}  # name -> Individual
        self.triples: Dict[tuple, Triple] = {}  # (s, p, o) -> Triple
        self.memberships: Dict[tuple, Membership] = {}  # (ind, cls) -> Membership
        self.attr_triples: Dict[
            tuple, AttributeTriple
        ] = {}  # (s, p, v) -> AttributeTriple

        # Set to track proofs we've already processed to avoid duplicate work
        # This is vital for proof trees that form a DAG (Directed Acyclic Graph)
        self.processed_proofs: Set[Proof] = set()

    def _register_individual(self, term: Term) -> None:
        """Adds an Individual to our store if it's not present."""
        if isinstance(term, Individual):
            if term.name not in self.individuals:
                self.individuals[term.name] = term

    def _add_atom_and_proof(self, atom: Atom, proof: Proof) -> None:
        """
        Converts a ground Atom into a KG fact (Triple, Membership, etc.)
        and adds it to our stores. If the fact already exists, it appends
        the proof to the fact's proof list.
        """
        if not atom.is_ground():
            # This should not happen if chainer is correct
            print(f"Warning: Skipping non-ground atom: {atom}")
            return

        s, p, o = atom.subject, atom.predicate, atom.object

        # Case 1: Class Membership (e.g., rdf:type(Ind_0, Person))
        if p == RDF.type and isinstance(o, Class):
            self._register_individual(s)
            key = (s.name, o.name)
            if key not in self.memberships:
                # Create a new Membership fact
                self.memberships[key] = Membership(
                    individual=s, cls=o, is_member=True, proofs=[]
                )
            # Add this proof to the fact's proof list
            self.memberships[key].proofs.append(proof)

        # Case 2: Relational Triple (e.g., parent(Ind_0, Ind_1))
        elif isinstance(o, Individual) and isinstance(p, Relation):
            self._register_individual(s)
            self._register_individual(o)
            key = (s.name, p.name, o.name)
            if key not in self.triples:
                # Create a new Triple fact
                self.triples[key] = Triple(
                    subject=s, predicate=p, object=o, positive=True, proofs=[]
                )
            # Add this proof to the fact's proof list
            self.triples[key].proofs.append(proof)

        # Case 3: Attribute Triple (e.g., hasAge(Ind_0, "30"))
        elif isinstance(p, Attribute):
            self._register_individual(s)
            # Note: object (o) is a LiteralValue here
            key = (s.name, p.name, o)
            if key not in self.attr_triples:
                # Create a new AttributeTriple fact
                self.attr_triples[key] = AttributeTriple(
                    subject=s, predicate=p, value=o, proofs=[]
                )
            # Add this proof to the fact's proof list
            self.attr_triples[key].proofs.append(proof)

        # Other atom types (e.g., variable predicates) are not handled

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

        for i, rule in enumerate(all_rules):
            print(f"\n[{i + 1}/{len(all_rules)}] Generating from rule: {rule.name}")

            # Generate all top-level proof trees starting from this rule
            proof_generator = self.chainer.generate_proof_trees(rule.name)

            top_level_proofs = 0
            for top_level_proof in proof_generator:
                top_level_proofs += 1

                # We traverse the entire proof tree (a DAG) from the top
                # and add every single atom (goal, intermediate, base)
                # to our KG.

                stack = [top_level_proof]
                while stack:
                    current_proof = stack.pop()

                    # If we've seen this exact proof object, skip
                    if current_proof in self.processed_proofs:
                        continue
                    self.processed_proofs.add(current_proof)

                    # Add the goal of this proof to our KG
                    self._add_atom_and_proof(current_proof.goal, current_proof)

                    # Add all sub-proofs to the stack to be processed
                    for sub_proof in current_proof.sub_proofs:
                        stack.append(sub_proof)

            if top_level_proofs == 0:
                print("  -> No proofs found (rule may not be a valid starting point).")
            else:
                print(f"  -> Found and processed {top_level_proofs} proof tree(s).")

        print("\n--- Data generation complete. ---")
        return self.build_knowledge_graph()

    def build_knowledge_graph(self) -> KnowledgeGraph:
        """
        Assembles the final KnowledgeGraph object from all collected data.
        """
        return KnowledgeGraph(
            # From schema
            attributes=list(self.schema_attributes.values()),
            classes=list(self.schema_classes.values()),
            relations=list(self.schema_relations.values()),
            # From generated data
            individuals=list(self.individuals.values()),
            triples=list(self.triples.values()),
            memberships=list(self.memberships.values()),
            attribute_triples=list(self.attr_triples.values()),
        )


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Ontology-based Knowledge Graph Data Generator"
    )
    parser.add_argument(
        "ontology_file",
        type=str,
        help="Path to the ontology file (e.g., 'my_ontology.ttl')",
    )
    parser.add_argument(
        "--max-recursion",
        type=int,
        default=2,
        help="Maximum depth for recursive rules (default: 2)",
    )
    args = parser.parse_args()

    try:
        # 1. Initialize Generator
        generator = KnowledgeGraphGenerator(
            args.ontology_file, max_recursion=args.max_recursion
        )

        # 2. Run Generation
        kg = generator.generate_data()

        # 3. Print Summary
        print("\n--- Knowledge Graph Generation Summary ---")
        print(f"  Schema Classes:    {len(kg.classes)}")
        print(f"  Schema Relations:  {len(kg.relations)}")
        print(f"  Schema Attributes: {len(kg.attributes)}")
        print("------------------------------------------")
        print(f"  Generated Individuals: {len(kg.individuals)}")
        print(f"  Generated Triples:     {len(kg.triples)}")
        print(f"  Generated Memberships: {len(kg.memberships)}")
        print(f"  Generated AttrTriples: {len(kg.attribute_triples)}")
        print("------------------------------------------")

        kg.print()

    except FileNotFoundError:
        print(
            f"Error: Ontology file not found at '{args.ontology_file}'", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred:\n{e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
