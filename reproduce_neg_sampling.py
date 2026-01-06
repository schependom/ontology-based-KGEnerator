
import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_structures import KnowledgeGraph, Triple, Individual, Relation, Class, Membership
from negative_sampler import NegativeSampler

def create_dummy_kg(n_triples=1000):
    """Create a dummy KG for testing."""
    kg = KnowledgeGraph(
        attributes=[],
        classes=[],
        relations=[],
        individuals=[],
        triples=[],
        memberships=[],
        attribute_triples=[]
    )
    
    # Create schema
    person_cls = Class(0, "Person")
    knows_rel = Relation(0, "knows")
    knows_rel.domain = {person_cls}
    knows_rel.range = {person_cls}
    
    schema_classes = {"Person": person_cls}
    schema_relations = {"knows": knows_rel}
    
    # Create individuals
    individuals = [Individual(i, f"p{i}") for i in range(100)]
    kg.individuals.extend(individuals)
    
    # Create triples
    for i in range(n_triples):
        s = individuals[i % 100]
        o = individuals[(i + 1) % 100]
        t = Triple(s, knows_rel, o, positive=True)
        kg.triples.append(t)
        
    return kg, schema_classes, schema_relations

def test_performance():
    print("Testing NegativeSampler performance...")
    kg, classes, relations = create_dummy_kg(n_triples=2000)
    
    sampler = NegativeSampler(classes, relations)
    
    start_time = time.time()
    # Use random strategy as baseline
    sampler.add_negative_samples(kg, strategy="random", ratio=1.0)
    end_time = time.time()
    
    print(f"Time taken for 2000 triples (random): {end_time - start_time:.4f}s")

def test_mixed_strategy():
    print("\nTesting Mixed Strategy...")
    kg, classes, relations = create_dummy_kg(n_triples=10)
    sampler = NegativeSampler(classes, relations)
    
    try:
        sampler.add_negative_samples(kg, strategy="mixed", ratio=1.0)
        print("Mixed strategy ran successfully.")
        
        # Check if we have negatives
        negatives = [t for t in kg.triples if not t.positive]
        print(f"Generated {len(negatives)} negative triples.")
        
    except ValueError as e:
        print(f"Caught expected error (if not implemented yet): {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

if __name__ == "__main__":
    test_performance()
    test_mixed_strategy()
