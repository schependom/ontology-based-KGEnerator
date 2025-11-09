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

| Argument                   | Description                                                   |
| -------------------------- | ------------------------------------------------------------- |
| `--ontology`               | Path to the ontology file                                     |
| `--seed`                   | Random seed for reproducibility                               |
| `--max_proof_depth`        | Maximum depth of the proof tree                               |
| `--max_nb_proofs_per_rule` | Number of proofs to try to prove each rule in the ontology    |
| `--reuse_prob`             | Probability $p_r$ of reusing an existing individual           |
| `--base_fact_prob`         | Probability $p_b$ of generating a base fact in the proof tree |

To see more information and the default values, run:

```bash
python src/generator.py --help
```
