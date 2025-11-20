# Ontology-based data generator for KGE models

Backward-chaining ontology-based data generator for Knowledge Graph Embedding models.

## Installation

Create a Python environment (tested with Python 3.11):

```bash
conda env create -f environment.yaml
conda activate KGEnerator
pip install -r requirements.txt
```

Note: The `graphviz` Python package requires the Graphviz system binary to render graphs. On macOS you can install it with:

```bash
brew install graphviz
```

## Usage

Basic usage (run from the project root). These scripts expect to be executed
from the `src/` directory as they import sibling modules:

```bash
# Visualize proofs (writes output to `output/`)
python src/proof_visualizer.py --ontology-path data/toy.ttl --output-dir output

# Generate datasets (CSV output)
python src/create_data.py --ontology data/family.ttl --output data/out
```

Work in progress as of 20/11/2025.
