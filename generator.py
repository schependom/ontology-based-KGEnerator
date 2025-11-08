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


def main():
    ontology_file = "pure-python/data/family2.ttl"

    # Parse the ontology
    try:
        parser = OntologyParser(ontology_file)
    except Exception as e:
        print(f"Error parsing ontology: {e}")
        return

    # Initialize the generator
    generator = BackwardChainingGenerator(parser)

    # Generate facts
    kg = generator.generate(max_depth=5, num_chains_per_rule=2)

    # 4. Print the results
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
    main()
