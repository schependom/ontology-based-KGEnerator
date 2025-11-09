"""
DESCRIPTION:

    Main script to generate facts using backward chaining on a given ontology.


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
    verbose: bool = False,
):
    """
    Main function to run the ontology-based data generator.

    Args:
        ontology_path (str): Path to the ontology file.
        seed (int): Random seed for reproducibility.
        max_proof_depth (int): Maximum depth for backward chaining.
        max_nb_proofs_per_rule (int): Number of diverse proofs to generate per rule.
        reuse_prob (float): Probability of reusing existing individuals.
        base_fact_prob (float): Base probability of generating base facts.
        verbose (bool): Enable verbose output.
    """

    # ------------------------------ PARSE ONTOLOGY ----------------------------- #

    print(f"Parsing ontology from: {ontology_path}")
    try:
        parsed_ont = OntologyParser(filepath=ontology_path)
    except Exception as e:
        print(f"Error parsing ontology: {e}")
        return

    # ------------------------------- GENERATE DATA ------------------------------ #

    print(f"\nInitializing generator with:")
    print(f"  - Seed: {seed}")
    print(f"  - Max proof depth: {max_proof_depth}")
    print(f"  - Proofs per rule: {max_nb_proofs_per_rule}")
    print(f"  - Reuse probability: {reuse_prob}")
    print(f"  - Base fact probability: {base_fact_prob}")
    print(f"  - Verbose: {verbose}")

    # Initialize the generator
    generator = BackwardChainingGenerator(
        parsed_ontology=parsed_ont,
        seed=seed,
        reuse_prob=reuse_prob,
        base_fact_prob=base_fact_prob,
        verbose=verbose,
    )

    # Generate facts
    kg = generator.generate(
        max_depth=max_proof_depth, num_proofs_per_rule=max_nb_proofs_per_rule
    )

    # ------------------------------- PRINT RESULTS ------------------------------ #

    print("\n" + "=" * 70)
    print("GENERATED FACTS")
    print("=" * 70)

    print("\n--- Base Facts ---")
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

    print("\n--- Inferred Facts ---")
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

    # Print comprehensive statistics
    generator.print_generation_statistics()


if __name__ == "__main__":
    #
    # ---------------------------------------------------------------------------- #
    #                               MAIN ENTRY POINT                               #
    # ---------------------------------------------------------------------------- #

    # ------------------------------ DEFAULT VALUES ------------------------------ #

    # Improved defaults based on analysis
    default_ontology = "data/family.ttl"
    default_seed = 42  # More standard seed
    default_max_proof_depth = 5  # Shallower to avoid excessive complexity
    default_max_nb_proofs_per_rule = 10  # Fewer but more diverse
    default_reuse_prob = 0.3  # Higher to create more connected graphs
    default_base_fact_prob = 0.15  # Higher to ensure base facts exist

    # ------------------------------ PARSE ARGUMENTS ----------------------------- #

    parser = argparse.ArgumentParser(
        description="Ontology-based Knowledge Graph Generator using Backward Chaining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python generator.py
  
  # Custom ontology with verbose output
  python generator.py --ontology my_ontology.ttl --verbose
  
  # Experiment with different probabilities
  python generator.py --reuse_prob 0.5 --base_fact_prob 0.2
  
  # Generate more diverse proofs per rule
  python generator.py --max_proofs_per_rule 20 --max_proof_depth 6
        """,
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
        help=f"Number of diverse proofs to generate per rule (default: {default_max_nb_proofs_per_rule})",
    )
    parser.add_argument(
        "--reuse_prob",
        type=float,
        default=default_reuse_prob,
        help=f"Probability of reusing an existing individual (default: {default_reuse_prob})",
    )
    parser.add_argument(
        "--base_fact_prob",
        type=float,
        default=default_base_fact_prob,
        help=f"Base probability of generating a base fact in the proof tree (default: {default_base_fact_prob})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging (default: False)",
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
        verbose=args.verbose,
    )
