# Ontology-based data generator for KGE models

Backward-chaining ontology-based data generator for Knowledge Graph Embedding models.

## Installation

Install the required packages using conda:

```bash
conda create -n KGEnerator python=3.14.0 -y -r requirements.txt
conda activate KGEnerator
```

## Usage

Run the generator script:

```bash
python src/generator.py
```

I'm working on adding command-line arguments for controlling

-   $p_r$: the probability of reusing an existing individual
-   $p_b$: the probability of generating a base fact in the proof tree
-   maximum depth of the proof tree
-   number of proofs to try to prove each rule in the ontology
