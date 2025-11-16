# Ontology-based data generator for KGE models

Backward-chaining ontology-based data generator for Knowledge Graph Embedding models.

## Installation

Create a Python environment:

```bash
conda env create -f environment.yaml
conda activate KGEnerator
```

## Usage

Run the generator script:

```bash
python src/generator.py <args>
```

Where `<args>` can include:

| Argument                | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| `--ontology-path`       | Path to the ontology file in turtle format.              |
| `--max-recursion-depth` | Maximum depth for recursive rule applications in proofs. |

To see more information and the default values, run:

```bash
python src/generator.py --help
```

### From and to CSV

```python
# Generate dataset
from rrn_dataset_generator import RRNDatasetGenerator

generator = RRNDatasetGenerator("family.ttl")
train_samples, test_samples = generator.generate_dataset(
    n_train=5000,
    n_test=500
)

# Save entire datasets
KnowledgeGraph.to_csv_batch(
    train_samples,
    output_dir="data/rrn_datasets/train",
    prefix="train_sample"
)

KnowledgeGraph.to_csv_batch(
    test_samples,
    output_dir="data/rrn_datasets/test",
    prefix="test_sample"
)

# Load entire datasets
train_samples = KnowledgeGraph.from_csv_batch(
    input_dir="data/rrn_datasets/train",
    prefix="train_sample"
)

test_samples = KnowledgeGraph.from_csv_batch(
    input_dir="data/rrn_datasets/test",
    prefix="test_sample",
    n_samples=100  # Load only first 100
)
```

Complete training workflow:

```bash
python rrn_dataset_generator.py \\
    --ontology data/family.ttl \\
    --output data/rrn_datasets/ \\
    --n-train 5000 \\
    --n-test 500
```

Then load and inspect the generated data:

```python
from data_structures import KnowledgeGraph

# Load training data
train_samples = KnowledgeGraph.from_csv_batch(
    input_dir="data/rrn_datasets/train",
    prefix="train_sample"
)

# Load test data
test_samples = KnowledgeGraph.from_csv_batch(
    input_dir="data/rrn_datasets/test",
    prefix="test_sample"
)

print(f"Loaded {len(train_samples)} training samples")
print(f"Loaded {len(test_samples)} test samples")

# Inspect a sample
sample_kg = train_samples[0]
print(f"\nSample 0 statistics:")
print(f"  Individuals: {len(sample_kg.individuals)}")
print(f"  Triples: {len(sample_kg.triples)}")
print(f"  Memberships: {len(sample_kg.memberships)}")

# Check positive/negative balance
pos_triples = [t for t in sample_kg.triples if t.positive]
neg_triples = [t for t in sample_kg.triples if not t.positive]
print(f"  Positive triples: {len(pos_triples)}")
print(f"  Negative triples: {len(neg_triples)}")
```

Verify the dataset:

```python
# Quick verification script
from data_structures import KnowledgeGraph
from pathlib import Path

def verify_dataset(dataset_dir: str, prefix: str):
    """Verify dataset integrity."""
    csv_files = list(Path(dataset_dir).glob(f"{prefix}_*.csv"))

    print(f"Found {len(csv_files)} CSV files")

    # Load a few samples
    for i, csv_file in enumerate(csv_files[:5]):
        kg = KnowledgeGraph.from_csv(str(csv_file))

        pos = len([t for t in kg.triples if t.positive])
        neg = len([t for t in kg.triples if not t.positive])

        print(f"\n{csv_file.name}:")
        print(f"  Individuals: {len(kg.individuals)}")
        print(f"  Classes: {len(kg.classes)}")
        print(f"  Relations: {len(kg.relations)}")
        print(f"  Triples: {len(kg.triples)} ({pos} pos, {neg} neg)")
        print(f"  Memberships: {len(kg.memberships)}")

        # Check balance
        if pos > 0 and neg > 0:
            ratio = neg / pos
            print(f"  Neg/Pos ratio: {ratio:.2f}")

# Run verification
verify_dataset("data/rrn_datasets/train", "train_sample")
verify_dataset("data/rrn_datasets/test", "test_sample")
```
