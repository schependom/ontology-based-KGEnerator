"""
DESCRIPTION

    KGE Model Train/Test Data Generator

    Generates independent knowledge graph samples for KGE model training.
    Each sample is a complete KG with unique individuals, base facts,
    derived inferences, and balanced positive/negative examples.

AUTHOR

    Vincent Van Schependom
"""

from collections import defaultdict
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import networkx as nx

from data_structures import KnowledgeGraph
from generate import KGenerator, extract_proof_map, atoms_to_knowledge_graph
from graph_visualizer import GraphVisualizer
from negative_sampler import NegativeSampler


class KGEDatasetGenerator:
    """
    Generates training and testing datasets for KGE models.

    Orchestrates:
    - Sample generation via KGenerator
    - Negative sampling via NegativeSampler
    - Train/test splitting
    - Validation and export
    """

    def __init__(
        self,
        ontology_file: str,
        max_recursion: int,
        global_max_depth: int,
        max_proofs_per_atom: int,
        individual_pool_size: int,
        individual_reuse_prob: float,
        neg_strategy: str,
        neg_ratio: float,
        neg_corrupt_base_facts: bool,
        verbose: bool,
        seed: Optional[int] = None,
    ):
        """
        Initialize dataset generator.

        Args:
            ontology_file: Path to .ttl ontology file
            max_recursion: Maximum depth for recursive rules
            global_max_depth: Hard limit on proof tree depth
            max_proofs_per_atom: Max proofs per atom
            individual_pool_size: Size of individual reuse pool
            individual_reuse_prob: Probability of reusing individuals
            neg_strategy: Negative sampling strategy
            neg_ratio: Ratio of negative to positive samples
            neg_corrupt_base_facts: Whether to corrupt base facts
            verbose: Enable detailed logging
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        self.verbose = verbose
        self.max_recursion_cap = max_recursion
        self.individual_pool_size = individual_pool_size
        self.individual_reuse_prob = individual_reuse_prob

        # Initialize KGenerator
        self.generator = KGenerator(
            ontology_file=ontology_file,
            max_recursion=max_recursion,
            global_max_depth=global_max_depth,
            max_proofs_per_atom=max_proofs_per_atom,
            individual_pool_size=individual_pool_size,
            individual_reuse_prob=individual_reuse_prob,
            verbose=False,  # Keep generator quiet during batch generation
            export_proof_visualizations=False,
        )

        # Store schema references
        self.schema_classes = self.generator.schema_classes
        self.schema_relations = self.generator.schema_relations
        self.schema_attributes = self.generator.schema_attributes
        self.rules = self.generator.parser.rules

        # Initialize NegativeSampler
        self.negative_sampler = NegativeSampler(
            schema_classes=self.schema_classes,
            schema_relations=self.schema_relations,
            domains=self.generator.parser.domains,
            ranges=self.generator.parser.ranges,
            verbose=verbose,
        )

        # Negative sampling config
        self.neg_strategy = neg_strategy
        self.neg_ratio = neg_ratio
        self.neg_corrupt_base_facts = neg_corrupt_base_facts

        # Track rule usage for coverage analysis
        self.train_rule_usage: Dict[str, int] = defaultdict(int)
        self.test_rule_usage: Dict[str, int] = defaultdict(int)

        print(f"Loaded {len(self.rules)} rules from ontology")
        print(
            f"Schema: {len(self.schema_classes)} classes, "
            f"{len(self.schema_relations)} relations, "
            f"{len(self.schema_attributes)} attributes"
        )
        print(f"Constraints: {len(self.generator.parser.constraints)}")

    def generate_dataset(
        self,
        n_train: int = 5,
        n_test: int = 2,
        min_individuals: int = 5,
        max_individuals: int = 30,
        min_rules_per_sample: int = 1,
        max_rules_per_sample: int = 7,
    ) -> tuple[List[KnowledgeGraph], List[KnowledgeGraph]]:
        """
        Generate complete training and testing datasets.

        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            min_individuals: Minimum individuals per sample
            max_individuals: Maximum individuals per sample
            min_rules_per_sample: Min rules to trigger per sample
            max_rules_per_sample: Max rules to trigger per sample

        Returns:
            Tuple of (train_samples, test_samples)
        """
        print(f"\n{'=' * 80}")
        print("GENERATING KGE DATASET")
        print(f"{'=' * 80}")
        print(f"Target: {n_train} train samples, {n_test} test samples")
        print(f"Individual range: {min_individuals}-{max_individuals}")
        print(f"Rules per sample: {min_rules_per_sample}-{max_rules_per_sample}")
        print(f"{'=' * 80}\n")

        # Generate training samples
        print("Generating training samples...")
        train_samples = self._generate_samples(
            n_samples=n_train,
            min_individuals=min_individuals,
            max_individuals=max_individuals,
            min_rules=min_rules_per_sample,
            max_rules=max_rules_per_sample,
            sample_type="TRAIN",
        )

        # Generate test samples (independent)
        print("\nGenerating test samples...")
        test_samples = self._generate_samples(
            n_samples=n_test,
            min_individuals=min_individuals,
            max_individuals=max_individuals,
            min_rules=min_rules_per_sample,
            max_rules=max_rules_per_sample,
            sample_type="TEST",
        )

        # Print summary
        self._print_dataset_summary(train_samples, test_samples)

        return train_samples, test_samples

    def _generate_samples(
        self,
        n_samples: int,
        min_individuals: int,
        max_individuals: int,
        min_rules: int,
        max_rules: int,
        sample_type: str,
    ) -> List[KnowledgeGraph]:
        """
        Generate a list of independent knowledge graph samples.

        Args:
            n_samples: Number of samples to generate
            min_individuals: Min individuals per sample
            max_individuals: Max individuals per sample
            min_rules: Min rules per sample
            max_rules: Max rules per sample
            sample_type: "TRAIN" or "TEST" (for logging)

        Returns:
            List of generated KG samples
        """
        samples = []
        failed_attempts = 0
        max_failed_attempts = n_samples * 10

        while len(samples) < n_samples and failed_attempts < max_failed_attempts:
            # Reset individual pool for each sample
            self.generator.chainer.reset_individual_pool()

            sample = self._generate_one_sample(
                min_individuals=min_individuals,
                max_individuals=max_individuals,
                min_rules=min_rules,
                max_rules=max_rules,
                sample_type=sample_type,
            )

            if sample is not None:
                samples.append(sample)
                if len(samples) % 100 == 0 or len(samples) == n_samples:
                    print(f"  [{sample_type}] Generated {len(samples)}/{n_samples}")
            else:
                failed_attempts += 1

        if len(samples) < n_samples:
            print(
                f"Warning: Only generated {len(samples)}/{n_samples} samples "
                f"after {failed_attempts} failed attempts"
            )

        return samples

    def _generate_one_sample(
        self,
        min_individuals: int,
        max_individuals: int,
        min_rules: int,
        max_rules: int,
        sample_type: str,
    ) -> Optional[KnowledgeGraph]:
        """
        Generate one complete, independent knowledge graph sample.

        Strategy:
        1. Randomly vary recursion depth (structural diversity)
        2. Randomly select subset of rules (content diversity)
        3. Generate proofs for selected rules
        4. Convert to KG
        5. Add negative samples via NegativeSampler

        Args:
            min_individuals: Minimum individuals required
            max_individuals: Maximum individuals allowed
            min_rules: Minimum rules to trigger
            max_rules: Maximum rules to trigger
            sample_type: "TRAIN" or "TEST"

        Returns:
            Generated KG sample, or None if generation failed
        """
        if not self.rules:
            return None

        # VARIANCE STRATEGY 1: Vary recursion depth
        current_recursion = random.randint(1, self.max_recursion_cap)
        self.generator.chainer.max_recursion_depth = current_recursion

        # VARIANCE STRATEGY 2: Random rule selection
        n_rules = random.randint(min_rules, min(max_rules, len(self.rules)))
        selected_rules = random.sample(self.rules, n_rules)

        # Track rule usage
        rule_usage = (
            self.train_rule_usage if sample_type == "TRAIN" else self.test_rule_usage
        )
        for rule in selected_rules:
            rule_usage[rule.name] += 1

        # Generate proofs and build proof map
        sample_proof_map = defaultdict(list)
        atoms_found = False

        for rule in selected_rules:
            proofs = self.generator.generate_proofs_for_rule(rule.name, max_proofs=None)
            if not proofs:
                continue

            # Select random subset of proofs
            max_proofs_to_merge = 3  # Keep small to control graph size
            n_select = random.randint(1, min(len(proofs), max_proofs_to_merge))
            selected = random.sample(proofs, n_select)

            for proof in selected:
                extracted_map = extract_proof_map(proof)
                for atom, proof_list in extracted_map.items():
                    sample_proof_map[atom].extend(proof_list)
                atoms_found = True

        if not atoms_found:
            return None

        # Convert to KG
        kg = atoms_to_knowledge_graph(
            atoms=set(sample_proof_map.keys()),
            schema_classes=self.schema_classes,
            schema_relations=self.schema_relations,
            schema_attributes=self.schema_attributes,
            proof_map=sample_proof_map,
        )

        # Validate size
        if not (min_individuals <= len(kg.individuals) <= max_individuals):
            return None

        # Add negatives via NegativeSampler
        kg = self.negative_sampler.add_negative_samples(
            kg,
            strategy=self.neg_strategy,
            ratio=self.neg_ratio,
            corrupt_base_facts=self.neg_corrupt_base_facts,
        )

        return kg

    @staticmethod
    def check_structural_isomorphism(kg1: KnowledgeGraph, kg2: KnowledgeGraph) -> bool:
        """
        Check if two KGs are structurally isomorphic.

        Ignores individual names but preserves:
        - Graph topology (relations)
        - Class memberships (node attributes)
        - Attribute values (node attributes)

        Args:
            kg1: First knowledge graph
            kg2: Second knowledge graph

        Returns:
            True if structurally isomorphic, False otherwise
        """

        def to_nx(kg):
            G = nx.MultiDiGraph()

            # Nodes with attributes
            for ind in kg.individuals:
                clss = frozenset(
                    [
                        m.cls.name
                        for m in kg.memberships
                        if m.individual == ind and m.is_member
                    ]
                )
                attrs = tuple(
                    sorted(
                        [
                            (at.predicate.name, str(at.value))
                            for at in kg.attribute_triples
                            if at.subject == ind
                        ]
                    )
                )
                G.add_node(ind.name, classes=clss, attrs=attrs)

            # Edges (relations)
            for t in kg.triples:
                if t.positive:
                    G.add_edge(t.subject.name, t.object.name, label=t.predicate.name)

            return G

        G1 = to_nx(kg1)
        G2 = to_nx(kg2)

        nm = nx.algorithms.isomorphism.categorical_node_match(
            ["classes", "attrs"], [frozenset(), tuple()]
        )
        em = nx.algorithms.isomorphism.categorical_edge_match("label", None)

        return nx.is_isomorphic(G1, G2, node_match=nm, edge_match=em)

    def _print_dataset_summary(
        self,
        train_samples: List[KnowledgeGraph],
        test_samples: List[KnowledgeGraph],
    ) -> None:
        """Print summary statistics for generated datasets."""

        def compute_stats(samples: List[KnowledgeGraph]) -> Dict:
            if not samples:
                return {}

            return {
                "n_samples": len(samples),
                "avg_individuals": sum(len(s.individuals) for s in samples)
                / len(samples),
                "avg_triples": sum(len(s.triples) for s in samples) / len(samples),
                "avg_pos_triples": sum(
                    sum(1 for t in s.triples if t.positive) for s in samples
                )
                / len(samples),
                "avg_neg_triples": sum(
                    sum(1 for t in s.triples if not t.positive) for s in samples
                )
                / len(samples),
                "avg_memberships": sum(len(s.memberships) for s in samples)
                / len(samples),
                "avg_pos_memberships": sum(
                    sum(1 for m in s.memberships if m.is_member) for s in samples
                )
                / len(samples),
                "avg_neg_memberships": sum(
                    sum(1 for m in s.memberships if not m.is_member) for s in samples
                )
                / len(samples),
            }

        train_stats = compute_stats(train_samples)
        test_stats = compute_stats(test_samples)

        print(f"\n{'=' * 80}")
        print("DATASET GENERATION COMPLETE")
        print(f"{'=' * 80}")

        # Check isomorphism
        print("Checking structural isomorphism...")
        isomorphic_count = 0
        for train_kg in train_samples:
            for test_kg in test_samples:
                if self.check_structural_isomorphism(train_kg, test_kg):
                    isomorphic_count += 1
                    break

        if isomorphic_count > 0:
            print(
                f"Warning: Found {isomorphic_count} isomorphic samples between train/test"
            )
        else:
            print("✓ No structural isomorphism between train and test")

        # Rule coverage
        print(f"\n--- Rule Coverage ---")
        print(f"Train: {len(self.train_rule_usage)}/{len(self.rules)} rules used")
        print(f"Test:  {len(self.test_rule_usage)}/{len(self.rules)} rules used")

        unused_in_train = set(r.name for r in self.rules) - set(
            self.train_rule_usage.keys()
        )
        unused_in_test = set(r.name for r in self.rules) - set(
            self.test_rule_usage.keys()
        )

        if unused_in_train:
            print(f"Warning: {len(unused_in_train)} rules unused in training")
        if unused_in_test:
            print(f"Warning: {len(unused_in_test)} rules unused in testing")

        print("\nTRAINING SET:")
        print(f"  Samples:           {train_stats.get('n_samples', 0)}")
        print(f"  Avg individuals:   {train_stats.get('avg_individuals', 0):.1f}")
        print(f"  Avg triples:       {train_stats.get('avg_triples', 0):.1f}")
        print(f"    - Positive:      {train_stats.get('avg_pos_triples', 0):.1f}")
        print(f"    - Negative:      {train_stats.get('avg_neg_triples', 0):.1f}")
        print(f"  Avg memberships:   {train_stats.get('avg_memberships', 0):.1f}")
        print(f"    - Positive:      {train_stats.get('avg_pos_memberships', 0):.1f}")
        print(f"    - Negative:      {train_stats.get('avg_neg_memberships', 0):.1f}")

        print("\nTEST SET:")
        print(f"  Samples:           {test_stats.get('n_samples', 0)}")
        print(f"  Avg individuals:   {test_stats.get('avg_individuals', 0):.1f}")
        print(f"  Avg triples:       {test_stats.get('avg_triples', 0):.1f}")
        print(f"    - Positive:      {test_stats.get('avg_pos_triples', 0):.1f}")
        print(f"    - Negative:      {test_stats.get('avg_neg_triples', 0):.1f}")
        print(f"  Avg memberships:   {test_stats.get('avg_memberships', 0):.1f}")
        print(f"    - Positive:      {test_stats.get('avg_pos_memberships', 0):.1f}")
        print(f"    - Negative:      {test_stats.get('avg_neg_memberships', 0):.1f}")

        print(f"{'=' * 80}\n")


# ============================================================================ #
#                         CSV SERIALIZATION METHODS                            #
# ============================================================================ #


def save_dataset_to_csv(
    samples: List[KnowledgeGraph],
    output_dir: str,
    prefix: str = "sample",
) -> None:
    """
    Save dataset to CSV files.

    Each sample saved as separate CSV with format:
        subject, predicate, object, label, fact_type

    Args:
        samples: List of KG samples
        output_dir: Directory to save files
        prefix: Prefix for file names
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(samples)} samples to {output_dir}/")

    for idx, kg in enumerate(samples):
        file_path = output_path / f"{prefix}_{idx:05d}.csv"
        kg.to_csv(str(file_path))

        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            print(f"  Saved {idx + 1}/{len(samples)}")

    print(f"Dataset saved to {output_dir}/")


def load_dataset_from_csv(
    input_dir: str,
    prefix: str = "sample",
    n_samples: Optional[int] = None,
) -> List[KnowledgeGraph]:
    """
    Load dataset from CSV files.

    Args:
        input_dir: Directory containing CSV files
        prefix: Prefix of files to load
        n_samples: Max samples to load (None = all)

    Returns:
        List of loaded KG samples
    """
    input_path = Path(input_dir)
    pattern = f"{prefix}_*.csv"
    csv_files = sorted(input_path.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {input_dir}/{pattern}")

    if n_samples is not None:
        csv_files = csv_files[:n_samples]

    print(f"Loading {len(csv_files)} samples from {input_dir}/")

    samples = []
    for idx, file_path in enumerate(csv_files):
        kg = KnowledgeGraph.from_csv(str(file_path))
        samples.append(kg)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(csv_files):
            print(f"  Loaded {idx + 1}/{len(csv_files)}")

    print(f"Dataset loaded from {input_dir}/")
    return samples


# ============================================================================ #
#                              MAIN ENTRY POINT                                #
# ============================================================================ #


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate KGE training/testing datasets from ontology"
    )
    parser.add_argument(
        "--ontology-path", type=str, required=True, help="Path to ontology file (.ttl)"
    )
    parser.add_argument(
        "--output", type=str, default="data/out/", help="Output directory"
    )
    parser.add_argument(
        "--n-train", type=int, default=5, help="Number of training samples"
    )
    parser.add_argument("--n-test", type=int, default=2, help="Number of test samples")
    parser.add_argument("--min-individuals", type=int, default=1)
    parser.add_argument("--max-individuals", type=int, default=1000)
    parser.add_argument("--max-recursion", type=int, default=10)
    parser.add_argument("--global-max-depth", type=int, default=10)
    parser.add_argument("--max-proofs-per-atom", type=int, default=10)
    parser.add_argument(
        "--individual-pool-size", type=int, default=50, help="Size of individual pool"
    )
    parser.add_argument(
        "--individual-reuse-prob",
        type=float,
        default=0,
        help="Probability of reusing individuals (0.0-1.0)",
    )
    parser.add_argument(
        "--neg-strategy",
        type=str,
        default="constrained",
        choices=["random", "constrained", "proof_based", "type_aware"],
        help="Negative sampling strategy",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive samples",
    )
    parser.add_argument(
        "--neg-corrupt-base-facts",
        action="store_true",
        help="For proof_based: corrupt base facts in proof trees",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--visualize", action="store_true", help="Generate graph visualizations"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = KGEDatasetGenerator(
        ontology_file=args.ontology_path,
        max_recursion=args.max_recursion,
        global_max_depth=args.global_max_depth,
        max_proofs_per_atom=args.max_proofs_per_atom,
        individual_pool_size=args.individual_pool_size,
        individual_reuse_prob=args.individual_reuse_prob,
        neg_strategy=args.neg_strategy,
        neg_ratio=args.neg_ratio,
        neg_corrupt_base_facts=args.neg_corrupt_base_facts,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Generate datasets
    train_samples, test_samples = generator.generate_dataset(
        n_train=args.n_train,
        n_test=args.n_test,
        min_individuals=args.min_individuals,
        max_individuals=args.max_individuals,
    )

    # Save to CSV
    save_dataset_to_csv(train_samples, f"{args.output}/train", prefix="train_sample")
    save_dataset_to_csv(test_samples, f"{args.output}/test", prefix="test_sample")

    print("\nDataset generation complete!")

    # Optional visualization
    if args.visualize:
        print("\nVisualizing samples...")
        visualizer = GraphVisualizer("train-test-graphs")

        for i, sample in enumerate(train_samples[:5]):  # Limit to 5
            visualizer.visualize(
                sample, f"train_sample_{i + 1}.png", title=f"TRAIN Sample {i + 1}"
            )

        for i, sample in enumerate(test_samples[:5]):  # Limit to 5
            visualizer.visualize(
                sample, f"test_sample_{i + 1}.png", title=f"TEST Sample {i + 1}"
            )


if __name__ == "__main__":
    main()
