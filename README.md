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

| Argument                   | Description                                                   | Default               |
| -------------------------- | ------------------------------------------------------------- | --------------------- |
| `--ontology`               | Path to the ontology file                                     | `../data/family2.ttl` |
| `--seed`                   | Random seed for reproducibility                               | `23`                  |
| `--max_proof_depth`        | Maximum depth of the proof tree                               | `10`                  |
| `--max_nb_proofs_per_rule` | Number of proofs to try to prove each rule in the ontology    | `10`                  |
| `--reuse_prob`             | Probability $p_r$ of reusing an existing individual           | `0.7`                 |
| `--base_fact_prob`         | Probability $p_b$ of generating a base fact in the proof tree | `0.3`                 |
