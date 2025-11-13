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
