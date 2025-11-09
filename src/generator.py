"""
DESCRIPTION:

    Main script to generate facts using backward chaining on a given ontology.

    It parses the ontology using parser.py and generates facts using chainer.py.
    It then prints both base facts and inferred facts.

AUTHOR

    Vincent Van Schependom
"""

from parser import OntologyParser
from chainer import BackwardChainingGenerator
from data_structures import AttributeTriple, Triple, Membership
import argparse


def main(
    ontology_path: str,
    seed: int,
    max_proof_depth: int,
    max_nb_proofs_per_rule: int,
    reuse_prob: float,
    base_fact_prob: float,
):
    """
    Main function to run the ontology-based data generator.

    Args:
        ontology_path (str): Path to the ontology file.
        seed (int): Random seed for reproducibility.
    """

    # ------------------------------ PARSE ONTOLOGY ----------------------------- #

    # Parse the ontology
    try:
        parsed_ont = OntologyParser(filepath=ontology_path)

    except Exception as e:
        print(f"Error parsing ontology: {e}")
        return

    # ------------------------------- GENERATE DATA ------------------------------ #

    # Initialize the generator
    generator = BackwardChainingGenerator(
        parsed_ontology=parsed_ont,
        seed=seed,
        reuse_prob=reuse_prob,
        base_fact_prob=base_fact_prob,
    )

    # Generate facts
    kg = generator.generate(
        max_depth=max_proof_depth, num_proofs_per_rule=max_nb_proofs_per_rule
    )

    # ------------------------------- PRINT RESULTS ------------------------------ #

    print("\n--- Generated Base Facts ---")
    base_facts = 0
    all_facts = kg.triples + kg.memberships + kg.attribute_triples
    for fact in all_facts:
        if not fact.is_inferred:
            base_facts += 1
            if isinstance(fact, Triple):
                print(
                    f"  {fact.subject.name} -{fact.predicate.name}-> {fact.object.name}"
                )
            elif isinstance(fact, Membership):
                print(f"  {fact.individual.name} rdf:type {fact.cls.name}")
            elif isinstance(fact, AttributeTriple):
                print(f"  {fact.subject.name} -{fact.predicate.name}-> {fact.value}")
    if base_facts == 0:
        print("  (No base facts generated)")

    print("\n--- Generated Inferred Facts ---")
    inferred_facts = 0
    for fact in all_facts:
        if fact.is_inferred:
            inferred_facts += 1
            if isinstance(fact, Triple):
                print(
                    f"  {fact.subject.name} -{fact.predicate.name}-> {fact.object.name}"
                )
            elif isinstance(fact, Membership):
                print(f"  {fact.individual.name} rdf:type {fact.cls.name}")
            elif isinstance(fact, AttributeTriple):
                print(f"  {fact.subject.name} -{fact.predicate.name}-> {fact.value}")
    if inferred_facts == 0:
        print("  (No inferred facts generated)")


if __name__ == "__main__":
    #
    # ---------------------------------------------------------------------------- #
    #                               MAIN ENTRY POINT                               #
    # ---------------------------------------------------------------------------- #

    # ------------------------------ DEFAULT VALUES ------------------------------ #

    default_ontology = "../data/family2.ttl"
    default_seed = 23
    default_max_proof_depth = 10
    default_max_nb_proofs_per_rule = 10
    default_reuse_prob = 0.7
    default_base_fact_prob = 0.3

    # ------------------------------ PARSE ARGUMENTS ----------------------------- #

    parser = argparse.ArgumentParser(
        description="Ontology-based Knowledge Graph Generator using Backward Chaining"
    )
    parser.add_argument(
        "--ontology",
        type=str,
        default=default_ontology,
        help=f"Path to the ontology file (default: {default_ontology})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help=f"Random seed for reproducibility (default: {default_seed})",
    )
    parser.add_argument(
        "--max_proof_depth",
        type=int,
        default=default_max_proof_depth,
        help=f"Maximum proof depth for backward chaining (default: {default_max_proof_depth})",
    )
    parser.add_argument(
        "--max_proofs_per_rule",
        type=int,
        default=default_max_nb_proofs_per_rule,
        help=f"Maximum number of proofs per rule (default: {default_max_nb_proofs_per_rule})",
    )
    parser.add_argument(
        "--reuse_prob",
        type=float,
        default=default_reuse_prob,
        help=f"Probability $p_r$ of reusing an existing individual (default: {default_reuse_prob})",
    )
    parser.add_argument(
        "--base_fact_prob",
        type=float,
        default=default_base_fact_prob,
        help=f"Probability $p_b$ of generating a base fact in the proof tree (default: {default_base_fact_prob})",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------------- #
    #                                   CALL MAIN                                  #
    # ---------------------------------------------------------------------------- #

    main(
        ontology_path=args.ontology,
        seed=args.seed,
        max_proof_depth=args.max_proof_depth,
        max_nb_proofs_per_rule=args.max_proofs_per_rule,
        reuse_prob=args.reuse_prob,
        base_fact_prob=args.base_fact_prob,
    )
