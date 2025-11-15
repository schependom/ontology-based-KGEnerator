"""
DESCRIPTION:

    Main script to generate facts using backward chaining on a given ontology.

    This is the orchestrator that ties together the parser and chainer to
    generate a complete knowledge graph with all possible derivable facts.

WORKFLOW:

    1. Parse the ontology using OntologyParser to extract:
       - Schema (classes, relations, attributes)
       - Rules (OWL 2 RL axioms converted to ExecutableRules)

    2. Initialize BackwardChainer with parsed rules

    3. For each rule in the ontology:
       - Generate all possible proof trees starting from that rule
       - Traverse each proof tree (depth-first) to extract all ground atoms

    4. Convert ground atoms into KnowledgeGraph facts:
       - Class memberships: (Individual, rdf:type, Class)
       - Relational triples: (Individual, Relation, Individual)
       - Attribute triples: (Individual, Attribute, LiteralValue)

    5. Store facts in KnowledgeGraph with deduplication:
       - Each fact tracks all proofs that derive it
       - Facts are deduplicated using unique keys

AUTHOR

    Vincent Van Schependom
"""

import sys
import argparse
import traceback
from typing import Dict, Set
from rdflib.namespace import RDF

# Custom imports
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

    Orchestrates the full data generation pipeline from ontology to
    complete knowledge graph with all derivable facts.

    The generator uses backward chaining to explore all possible proof
    paths in the ontology, generating individuals and facts along the way.
    """

    def __init__(self, ontology_file: str, max_recursion: int = 2):
        """
        Initializes the parser and chainer.

        Args:
            ontology_file (str): Path to the .ttl ontology file.
            max_recursion (int): The maximum depth for recursive rules.
                                 This prevents infinite recursion in rules like:
                                 parent(X,Y) ∧ parent(Y,Z) → ancestor(X,Z)
        """
        print(f"Loading and parsing ontology from: {ontology_file}")

        # Parse the ontology to extract schema and rules
        self.parser = OntologyParser(ontology_file)

        # Initialize the backward chainer with all rules
        self.chainer = BackwardChainer(
            all_rules=self.parser.rules, max_recursion_depth=max_recursion
        )

        # Store schemas from the parser
        self.schema_classes = self.parser.classes
        self.schema_relations = self.parser.relations
        self.schema_attributes = self.parser.attributes

        # Initialize storage for generated data
        # We use dictionaries with unique keys for automatic deduplication
        self.individuals: Dict[str, Individual] = {}  # name -> Individual
        self.triples: Dict[tuple, Triple] = {}  # (subj, pred, obj) -> Triple
        self.memberships: Dict[tuple, Membership] = {}  # (ind, cls) -> Membership
        self.attr_triples: Dict[
            tuple, AttributeTriple
        ] = {}  # (subj, pred, val) -> AttributeTriple

        # Track processed proofs to avoid duplicate work
        # This is critical because proof trees form a DAG structure
        # Example: If proofs A→B and A→C both depend on D, we only process D once
        self.processed_proofs: Set[Proof] = set()

    def _register_individual(self, term: Term) -> None:
        """
        Adds an Individual to our store if it's not already present.

        This ensures each individual is only stored once, even if it appears
        in multiple facts.

        Args:
            term (Term): A term that should be an Individual.
        """
        if isinstance(term, Individual):
            if term.name not in self.individuals:
                self.individuals[term.name] = term
        else:
            print(f"Warning: Expected Individual term, got: {term}")

    def _add_atom_and_proof(self, atom: Atom, proof: Proof) -> None:
        """
        Converts a ground Atom into a KG fact and adds it to our stores.

        If the fact already exists, appends this proof to its proof list.
        This allows tracking multiple derivation paths for the same fact.

        Args:
            atom (Atom):  The ground atom to add (must be fully ground).
            proof (Proof): The proof object that derived this atom.
        """
        # Sanity check: goal of a proof must always be ground
        if not atom.is_ground():
            print(f"Warning: Skipping non-ground atom: {atom}")
            return

        # Extract components of the atom
        s, p, o = atom.subject, atom.predicate, atom.object

        # ==================== CLASS MEMBERSHIPS ==================== #
        # Pattern: (Individual, rdf:type, Class)
        if p == RDF.type and isinstance(o, Class):
            self._register_individual(s)
            key = (s.name, o.name)
            if key not in self.memberships:
                # Create new membership fact
                self.memberships[key] = Membership(
                    individual=s, cls=o, is_member=True, proofs=[]
                )
            # Append this proof to the fact's proof list
            self.memberships[key].proofs.append(proof)

        # ==================== RELATIONAL TRIPLES ==================== #
        # Pattern: (Individual, Relation, Individual)
        elif isinstance(o, Individual) and isinstance(p, Relation):
            self._register_individual(s)
            self._register_individual(o)
            key = (s.name, p.name, o.name)
            if key not in self.triples:
                # Create new relational triple
                self.triples[key] = Triple(
                    subject=s, predicate=p, object=o, positive=True, proofs=[]
                )
            # Append this proof to the fact's proof list
            self.triples[key].proofs.append(proof)

        # ==================== ATTRIBUTE TRIPLES ==================== #
        # Pattern: (Individual, Attribute, LiteralValue)
        elif isinstance(p, Attribute):
            self._register_individual(s)
            # Note: object (o) is a LiteralValue (string, int, etc.)
            key = (s.name, p.name, o)
            if key not in self.attr_triples:
                # Create new attribute triple
                self.attr_triples[key] = AttributeTriple(
                    subject=s, predicate=p, value=o, proofs=[]
                )
            # Append this proof to the fact's proof list
            self.attr_triples[key].proofs.append(proof)

        # Other atom patterns (e.g., variable predicates) are not handled

    def generate_data(self) -> KnowledgeGraph:
        """
        Runs the full generation process.

        This method:
        1. Iterates through all rules in the ontology
        2. Generates all possible proofs starting from each rule
        3. Traverses each proof tree to extract atoms
        4. Converts atoms to KG facts and stores them

        Returns:
            KnowledgeGraph: The complete generated knowledge graph.
        """
        all_rules = self.parser.rules

        # Early exit if no rules
        if not all_rules:
            print("Warning: No rules found in the ontology. No data will be generated.")
            return self.build_knowledge_graph()

        print(f"Starting data generation from {len(all_rules)} rules...")

        # Iterate through all rules as starting points
        for i, rule in enumerate(all_rules):
            print(f"\n[{i + 1}/{len(all_rules)}] Generating from rule: {rule.name}")

            # Generate all top-level proof trees starting from this rule
            proof_generator = self.chainer.generate_proof_trees(rule.name)
            top_level_proofs = 0

            # Process each top-level proof
            for top_level_proof in proof_generator:
                top_level_proofs += 1

                # Traverse the entire proof tree (DAG) from the top
                # We use depth-first traversal with a stack
                stack = [top_level_proof]

                while stack:
                    current_proof = stack.pop()

                    # Skip if we've already processed this proof
                    if current_proof in self.processed_proofs:
                        continue

                    # Mark as processed
                    self.processed_proofs.add(current_proof)

                    # Add the goal of this proof to our KG
                    self._add_atom_and_proof(current_proof.goal, current_proof)

                    # Add all sub-proofs to the stack for processing
                    for sub_proof in current_proof.sub_proofs:
                        stack.append(sub_proof)

            if top_level_proofs == 0:
                print("  -> No proofs found (rule may not be a valid starting point).")
            else:
                print(f"  -> Found and processed {top_level_proofs} proof tree(s).")

        print("\n--- Data generation complete. ---")

        # Return the assembled knowledge graph
        return self.build_knowledge_graph()

    def build_knowledge_graph(self) -> KnowledgeGraph:
        """
        Assembles the final KnowledgeGraph object from all collected data.

        Returns:
            KnowledgeGraph: Complete knowledge graph with schema and generated data.
        """
        return KnowledgeGraph(
            # Schema elements from parser
            attributes=list(self.schema_attributes.values()),
            classes=list(self.schema_classes.values()),
            relations=list(self.schema_relations.values()),
            # Generated data from chainer
            individuals=list(self.individuals.values()),
            triples=list(self.triples.values()),
            memberships=list(self.memberships.values()),
            attribute_triples=list(self.attr_triples.values()),
        )


def main():
    """
    Main entry point for the script.

    Parses command-line arguments and runs the generation pipeline.
    """
    # ==================== PARSE ARGUMENTS ==================== #

    default_ontology_path = "data/family.ttl"
    default_max_recursion = 4

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
    args = parser.parse_args()

    # ==================== RUN GENERATION PIPELINE ==================== #

    try:
        # Initialize Generator
        generator = KGenerator(args.ontology_path, max_recursion=args.max_recursion)

        # Run Generation
        kg = generator.generate_data()

        # Print Summary
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

        # Print full knowledge graph
        kg.print()

    # ==================== ERROR HANDLING ==================== #

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
