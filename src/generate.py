"""
DESCRIPTION:

    Knowledge Graph Generator (KGenerator).

    Core class for generating synthetic knowledge graphs from ontologies using
    backward chaining. Handles proof generation and fact collection.

AUTHOR

    Vincent Van Schependom
"""

import sys
import argparse
import traceback
from typing import Dict, Set, List, Optional
from collections import defaultdict
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
from negative_sampler import NegativeSampler


# ============================================================================ #
#                         SHARED UTILITY FUNCTIONS                             #
# ============================================================================ #


def extract_proof_map(proof: Proof) -> Dict[Atom, List[Proof]]:
    """
    Recursively extracts atoms and maps them to the proofs that derive them.

    Unlike extract_all_atoms_from_proof, this preserves Proof objects,
    allowing determination of whether facts are base or derived.

    Args:
        proof: Root proof to extract from

    Returns:
        Mapping from atoms to all proofs that derive them
    """
    proof_map = defaultdict(list)

    # Map goal of this proof node
    proof_map[proof.goal].append(proof)

    # Recursively collect from sub-proofs
    for sub_proof in proof.sub_proofs:
        sub_map = extract_proof_map(sub_proof)
        for atom, proofs in sub_map.items():
            proof_map[atom].extend(proofs)

    return proof_map


def extract_all_atoms_from_proof(proof: Proof) -> Set[Atom]:
    """
    Recursively extracts ALL atoms from a proof tree via DFS traversal.

    Includes:
    - The goal atom (conclusion)
    - All atoms from sub-proofs (premises)
    - Atoms at all levels (base facts + intermediate inferences)

    Args:
        proof: The proof tree to extract atoms from

    Returns:
        Set of all ground atoms in the proof tree
    """
    atoms: Set[Atom] = set()
    atoms.add(proof.goal)

    for sub_proof in proof.sub_proofs:
        atoms.update(extract_all_atoms_from_proof(sub_proof))

    return atoms


def atoms_to_knowledge_graph(
    atoms: Set[Atom],
    schema_classes: Dict[str, Class],
    schema_relations: Dict[str, Relation],
    schema_attributes: Dict[str, Attribute],
    proof_map: Optional[Dict[Atom, List[Proof]]] = None,
) -> KnowledgeGraph:
    """
    Converts a set of ground atoms into a KnowledgeGraph.

    Organizes atoms into:
    - Memberships: (Individual, rdf:type, Class)
    - Triples: (Individual, Relation, Individual)
    - AttributeTriples: (Individual, Attribute, LiteralValue)

    All facts are positive at this stage. Negatives added separately.

    Args:
        atoms: Set of ground atoms to convert
        schema_classes: Schema classes from parser
        schema_relations: Schema relations from parser
        schema_attributes: Schema attributes from parser
        proof_map: Optional mapping from atoms to their proofs

    Returns:
        Organized knowledge graph structure
    """
    individuals: Dict[str, Individual] = {}
    triples: Dict[tuple, Triple] = {}
    memberships: Dict[tuple, Membership] = {}
    attr_triples: Dict[tuple, AttributeTriple] = {}

    def register_individual(term) -> None:
        """Register an individual if not seen before."""
        if isinstance(term, Individual):
            if term.name not in individuals:
                individuals[term.name] = term

    # Convert each atom to appropriate KG structure
    for atom in atoms:
        s, p, o = atom.subject, atom.predicate, atom.object

        # Get proofs for this atom if available
        current_proofs = []
        if proof_map and atom in proof_map:
            current_proofs = proof_map[atom]

        # CLASS MEMBERSHIPS
        if p == RDF.type and isinstance(o, Class):
            register_individual(s)
            key = (s.name, o.name)
            if key not in memberships:
                memberships[key] = Membership(s, o, True, proofs=[])
            if current_proofs:
                memberships[key].proofs.extend(current_proofs)

        # RELATIONAL TRIPLES
        elif isinstance(o, Individual) and isinstance(p, Relation):
            register_individual(s)
            register_individual(o)
            key = (s.name, p.name, o.name)
            if key not in triples:
                triples[key] = Triple(s, p, o, True, proofs=[])
            if current_proofs:
                triples[key].proofs.extend(current_proofs)

        # ATTRIBUTES
        elif isinstance(p, Attribute):
            register_individual(s)
            key = (s.name, p.name, o)
            if key not in attr_triples:
                attr_triples[key] = AttributeTriple(s, p, o, proofs=[])
            if current_proofs:
                attr_triples[key].proofs.extend(current_proofs)

    return KnowledgeGraph(
        attributes=list(schema_attributes.values()),
        classes=list(schema_classes.values()),
        relations=list(schema_relations.values()),
        individuals=list(individuals.values()),
        triples=list(triples.values()),
        memberships=list(memberships.values()),
        attribute_triples=list(attr_triples.values()),
    )


# ============================================================================ #
#                           CORE GENERATOR CLASS                               #
# ============================================================================ #


class KGenerator:
    """
    Knowledge Graph Generator.

    Orchestrates proof generation via backward chaining for synthetic
    knowledge graph creation. Designed to be called by create_data.py
    for train/test split generation.
    """

    def __init__(
        self,
        ontology_file: str,
        max_recursion: int,
        global_max_depth: int,
        max_proofs_per_atom: int,
        individual_pool_size: int,
        individual_reuse_prob: float,
        verbose: bool = False,
        export_proof_visualizations: bool = False,
    ):
        """
        Initialize generator with ontology and generation parameters.

        Args:
            ontology_file: Path to .ttl ontology file
            max_recursion: Maximum depth for recursive rules
            global_max_depth: Hard limit on total proof tree depth
            max_proofs_per_atom: Max proofs per atom (None = unlimited)
            individual_pool_size: Size of individual reuse pool
            individual_reuse_prob: Probability of reusing vs creating individuals
            verbose: Enable detailed logging
            export_proof_visualizations: Export proof trees for verification
        """
        self.verbose = verbose
        self.export_proof_visualizations = export_proof_visualizations

        # Parse ontology
        self.parser = OntologyParser(ontology_file)

        # Initialize backward chainer with constraints
        self.chainer = BackwardChainer(
            all_rules=self.parser.rules,
            constraints=self.parser.constraints,
            max_recursion_depth=max_recursion,
            global_max_depth=global_max_depth,
            max_proofs_per_atom=max_proofs_per_atom,
            individual_pool_size=individual_pool_size,
            individual_reuse_prob=individual_reuse_prob,
            verbose=verbose,
            export_proof_visualizations=export_proof_visualizations,
        )

        # Store schema references
        self.schema_classes = self.parser.classes
        self.schema_relations = self.parser.relations
        self.schema_attributes = self.parser.attributes

    def generate_proofs_for_rule(
        self,
        rule_name: str,
        max_proofs: Optional[int] = None,
    ) -> List[Proof]:
        """
        Generate proof trees for a specific rule.

        This is the main method called by create_data.py for sample generation.

        Args:
            rule_name: Name of the rule to generate proofs for
            max_proofs: Maximum number of proofs to generate (None = unlimited)

        Returns:
            List of valid, constraint-checked proof trees
        """
        if rule_name not in self.chainer.all_rules:
            if self.verbose:
                print(f"Warning: Rule '{rule_name}' not found")
            return []

        proofs = []
        proof_generator = self.chainer.generate_proof_trees(rule_name)

        try:
            for i, proof in enumerate(proof_generator):
                if max_proofs is not None and i >= max_proofs:
                    break
                proofs.append(proof)

        except Exception as e:
            print(f"Error generating proofs for {rule_name}: {e}")
            if self.verbose:
                traceback.print_exc()

        return proofs

    def generate_full_graph(self) -> KnowledgeGraph:
        """
        Generate complete knowledge graph with ALL derivable facts.

        This method is primarily for testing and verification. For train/test
        generation, use create_data.py which calls generate_proofs_for_rule()
        for controlled sample generation.

        NOTE: Output can be extremely large depending on ontology complexity.

        Returns:
            Complete knowledge graph with all positive facts
        """
        print("Generating complete knowledge graph from all rules...")
        print(
            f"Rules: {len(self.parser.rules)}, Constraints: {len(self.parser.constraints)}"
        )

        all_rules = self.parser.rules
        if not all_rules:
            print("Warning: No rules found in ontology")
            return self._build_empty_kg()

        # Storage for facts with proof tracking
        individuals: Dict[str, Individual] = {}
        triples: Dict[tuple, Triple] = {}
        memberships: Dict[tuple, Membership] = {}
        attr_triples: Dict[tuple, AttributeTriple] = {}
        processed_proofs: Set[Proof] = set()

        # Track statistics
        stats = {"proofs_accepted": 0, "rules_without_proofs": 0}

        # Generate proofs from each rule as starting point
        for i, rule in enumerate(all_rules):
            if self.verbose:
                print(f"\n[{i + 1}/{len(all_rules)}] Processing rule: {rule.name}")

            proof_generator = self.chainer.generate_proof_trees(rule.name)
            top_level_proofs = 0

            for top_level_proof in proof_generator:
                top_level_proofs += 1
                stats["proofs_accepted"] += 1

                # DFS traversal of proof tree (DAG)
                stack = [top_level_proof]
                while stack:
                    current_proof = stack.pop()

                    if current_proof in processed_proofs:
                        continue
                    processed_proofs.add(current_proof)

                    # Add goal to KG
                    self._add_atom_and_proof(
                        current_proof.goal,
                        current_proof,
                        individuals,
                        triples,
                        memberships,
                        attr_triples,
                    )

                    # Queue sub-proofs
                    stack.extend(current_proof.sub_proofs)

            if top_level_proofs == 0:
                stats["rules_without_proofs"] += 1
                if self.verbose:
                    print(f"  No valid proofs for {rule.name}")

        print(f"\nGeneration complete:")
        print(f"  Proofs accepted: {stats['proofs_accepted']}")
        print(
            f"  Rules without proofs: {stats['rules_without_proofs']}/{len(all_rules)}"
        )

        return self._build_kg(individuals, triples, memberships, attr_triples)

    def _add_atom_and_proof(
        self,
        atom: Atom,
        proof: Proof,
        individuals: Dict[str, Individual],
        triples: Dict[tuple, Triple],
        memberships: Dict[tuple, Membership],
        attr_triples: Dict[tuple, AttributeTriple],
    ) -> None:
        """
        Convert a ground atom into a KG fact and add to storage.

        If fact exists, appends proof to track multiple derivation paths.

        Args:
            atom: Ground atom to add
            proof: Proof object that derived this atom
            individuals: Storage for individuals
            triples: Storage for relational triples
            memberships: Storage for class memberships
            attr_triples: Storage for attribute triples
        """
        if not atom.is_ground():
            if self.verbose:
                print(f"Warning: Skipping non-ground atom: {atom}")
            return

        s, p, o = atom.subject, atom.predicate, atom.object

        # CLASS MEMBERSHIPS
        if p == RDF.type and isinstance(o, Class):
            self._register_individual(s, individuals)
            key = (s.name, o.name)
            if key not in memberships:
                memberships[key] = Membership(s, o, True, proofs=[])
            memberships[key].proofs.append(proof)

        # RELATIONAL TRIPLES
        elif isinstance(o, Individual) and isinstance(p, Relation):
            self._register_individual(s, individuals)
            self._register_individual(o, individuals)
            key = (s.name, p.name, o.name)
            if key not in triples:
                triples[key] = Triple(s, p, o, True, proofs=[])
            triples[key].proofs.append(proof)

        # ATTRIBUTE TRIPLES
        elif isinstance(p, Attribute):
            self._register_individual(s, individuals)
            key = (s.name, p.name, o)
            if key not in attr_triples:
                attr_triples[key] = AttributeTriple(s, p, o, proofs=[])
            attr_triples[key].proofs.append(proof)

    def _register_individual(
        self, term: Term, individuals: Dict[str, Individual]
    ) -> None:
        """Add individual to storage if not present."""
        if isinstance(term, Individual) and term.name not in individuals:
            individuals[term.name] = term

    def _build_kg(
        self,
        individuals: Dict[str, Individual],
        triples: Dict[tuple, Triple],
        memberships: Dict[tuple, Membership],
        attr_triples: Dict[tuple, AttributeTriple],
    ) -> KnowledgeGraph:
        """Assemble final KnowledgeGraph from collected data."""
        return KnowledgeGraph(
            attributes=list(self.schema_attributes.values()),
            classes=list(self.schema_classes.values()),
            relations=list(self.schema_relations.values()),
            individuals=list(individuals.values()),
            triples=list(triples.values()),
            memberships=list(memberships.values()),
            attribute_triples=list(attr_triples.values()),
        )

    def _build_empty_kg(self) -> KnowledgeGraph:
        """Create empty knowledge graph with schema only."""
        return KnowledgeGraph(
            attributes=list(self.schema_attributes.values()),
            classes=list(self.schema_classes.values()),
            relations=list(self.schema_relations.values()),
            individuals=[],
            triples=[],
            memberships=[],
            attribute_triples=[],
        )


# ============================================================================ #
#                         COMMAND-LINE INTERFACE                               #
# ============================================================================ #


def main():
    """
    Main entry point for full graph generation.

    For train/test split generation, use create_data.py instead.
    This script is for verification and testing on small ontologies.
    """
    parser = argparse.ArgumentParser(
        description="Generate complete knowledge graph from ontology (for verification)"
    )
    parser.add_argument(
        "--ontology-path",
        type=str,
        default="data/toy.ttl",
        help="Path to ontology file (.ttl)",
    )
    parser.add_argument(
        "--max-recursion",
        type=int,
        default=3,
        help="Maximum depth for recursive rules",
    )
    parser.add_argument(
        "--global-max-depth",
        type=int,
        default=10,
        help="Hard limit on total proof tree depth",
    )
    parser.add_argument(
        "--max-proofs-per-atom",
        type=int,
        default=5,
        help="Max proofs per atom (None = unlimited)",
    )
    parser.add_argument(
        "--individual-pool-size",
        type=int,
        default=50,
        help="Size of individual pool",
    )
    parser.add_argument(
        "--individual-reuse-prob",
        type=float,
        default=0,
        help="Probability of reusing individuals",
    )
    parser.add_argument(
        "--export-proofs",
        action="store_true",
        help="Export all proof trees for verification",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--neg-strategy",
        type=str,
        default=None,
        choices=["random", "constrained", "proof_based", "type_aware", "mixed"],
        help="Negative sampling strategy (optional)",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive samples",
    )
    parser.add_argument(
        "--corrupt-base-facts",
        action="store_true",
        help="Corrupt base facts for proof-based strategy",
    )
    args = parser.parse_args()

    try:
        # Initialize generator
        generator = KGenerator(
            ontology_file=args.ontology_path,
            max_recursion=args.max_recursion,
            global_max_depth=args.global_max_depth,
            max_proofs_per_atom=args.max_proofs_per_atom,
            individual_pool_size=args.individual_pool_size,
            individual_reuse_prob=args.individual_reuse_prob,
            verbose=args.verbose,
            export_proof_visualizations=args.export_proofs,
        )

        # Generate full graph
        kg = generator.generate_full_graph()

        # Add negative samples if requested
        if args.neg_strategy and args.neg_ratio > 0:
            print(f"\nAdding negative samples (Strategy: {args.neg_strategy}, Ratio: {args.neg_ratio})...")
            
            # Initialize NegativeSampler
            # We need schema info which is in the generator
            sampler = NegativeSampler(
                schema_classes=generator.schema_classes,
                schema_relations=generator.schema_relations,
                domains=generator.parser.domains,
                ranges=generator.parser.ranges,
                verbose=args.verbose,
            )
            
            kg = sampler.add_negative_samples(
                kg,
                strategy=args.neg_strategy,
                ratio=args.neg_ratio,
                corrupt_base_facts=args.corrupt_base_facts,
                export_proofs=args.export_proofs,
                output_dir="proof-trees" if args.export_proofs else None,
            )

        # check if the kg is not too big
        if len(kg.triples) + len(kg.memberships) > 100:
            print("Warning: Generated knowledge graph is very large (>10,000 facts).")
            print("Not saving visualization to avoid performance issues.")
        else:
            kg.save_visualization(
                output_path=".",
                output_name="full_knowledge_graph",
                title="Complete Knowledge Graph",
            )

        # Print summary
        print("\n--- Knowledge Graph Summary ---")
        print(f"  Schema:")
        print(f"    Classes:    {len(kg.classes)}")
        print(f"    Relations:  {len(kg.relations)}")
        print(f"    Attributes: {len(kg.attributes)}")
        print(f"  Generated Data:")
        print(f"    Individuals: {len(kg.individuals)}")
        print(f"    Triples:     {len(kg.triples)}")
        print(f"    Memberships: {len(kg.memberships)}")
        print(f"    Attributes:  {len(kg.attribute_triples)}")
        print("-------------------------------")

        if args.verbose:
            kg.print()

        if args.export_proofs:
            print(f"\nProof trees exported to: proof-trees/")

    except FileNotFoundError:
        print(
            f"Error: Ontology file not found at '{args.ontology_path}'", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
