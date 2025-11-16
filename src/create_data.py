"""
DESCRIPTION

    KGE model Train/Test Data Generator

    This script generates independent knowledge graph samples suitable for
    training KGE models. Each sample is a complete knowledge graph with:
        - Unique individuals (different per sample)
        - Mix of base facts and derived inferences
        - Positive and negative triples
        - All ground atoms (no variables)

WORKFLOW

    1. Initialize KGenerator with ontology
    2. Generate N training samples where each sample:
       - Randomly selects starting rules
       - Generates proof trees via backward chaining
       - Extracts ALL atoms from proof trees (base + inferred)
       - Converts atoms to KG facts using shared utilities
       - Generates negative samples via local CWA
    3. Generate M test samples using same process
    4. Save datasets to CSV files for KGE training

AUTHOR

    Vincent Van Schependom
"""

import csv
import random
import argparse
from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict
from collections import defaultdict

# Custom imports
from data_structures import (
    KnowledgeGraph,
    Triple,
    Membership,
    AttributeTriple,
    Individual,
    Class,
    Relation,
    Attribute,
    Atom,
    Proof,
)


from generator import (
    KGenerator,
    extract_all_atoms_from_proof,
    atoms_to_knowledge_graph,
)


class KGEDatasetGenerator:
    """
    Generates training and testing datasets for KGE models.
    """

    def __init__(
        self,
        ontology_file: str,
        max_recursion: int = 2,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initializes the dataset generator.

        Args:
            ontology_file (str):    Path to the .ttl ontology file.
            max_recursion (int):    Maximum depth for recursive rules.
            seed (Optional[int]):   Random seed for reproducibility.
            verbose (bool):         Enable detailed logging.
        """
        if seed is not None:
            random.seed(seed)

        self.verbose = verbose
        self.ontology_file = ontology_file

        # REFACTORED: Use KGenerator instead of creating parser/chainer directly
        if self.verbose:
            print(f"Initializing KGenerator from: {ontology_file}")

        self.generator = KGenerator(
            ontology_file=ontology_file,
            max_recursion=max_recursion,
            verbose=False,  # Keep generator quiet during batch generation
        )

        # Store references for convenience
        self.chainer = self.generator.chainer
        self.parser = self.generator.parser
        self.schema_classes = self.generator.schema_classes
        self.schema_relations = self.generator.schema_relations
        self.schema_attributes = self.generator.schema_attributes
        self.rules = self.parser.rules

        if self.verbose:
            print(f"Loaded {len(self.rules)} rules from ontology")
            print(
                f"Schema: {len(self.schema_classes)} classes, "
                f"{len(self.schema_relations)} relations, "
                f"{len(self.schema_attributes)} attributes"
            )
            print(f"Constraints: {len(self.parser.constraints)}")

    def generate_dataset(
        self,
        n_train: int = 5,
        n_test: int = 2,
        min_individuals: int = 5,
        max_individuals: int = 30,
        min_rules_per_sample: int = 1,
        max_rules_per_sample: int = 7,
    ) -> Tuple[List[KnowledgeGraph], List[KnowledgeGraph]]:
        """
        Generates complete training and testing datasets.

        Args:
            n_train (int):                  Number of training samples.
            n_test (int):                   Number of test samples.
            min_individuals (int):          Minimum individuals per sample.
            max_individuals (int):          Maximum individuals per sample.
            min_rules_per_sample (int):     Min rules to trigger per sample.
            max_rules_per_sample (int):     Max rules to trigger per sample.

        Returns:
            Tuple[List[KG], List[KG]]: Training and testing samples.
        """
        print(f"\n{'=' * 80}")
        print("GENERATING KGE DATASET")
        print(f"{'=' * 80}")
        print(f"Target: {n_train} train samples, {n_test} test samples")
        print(f"Individual range: {min_individuals}-{max_individuals} per sample")
        print(f"Rules per sample: {min_rules_per_sample}-{max_rules_per_sample}")
        print(f"Constraints: {len(self.parser.constraints)} active")
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

        # Generate test samples (completely independent)
        print("\nGenerating test samples...")
        test_samples = self._generate_samples(
            n_samples=n_test,
            min_individuals=min_individuals,
            max_individuals=max_individuals,
            min_rules=min_rules_per_sample,
            max_rules=max_rules_per_sample,
            sample_type="TEST",
        )

        # Print summary statistics
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
        Generates a list of independent knowledge graph samples.

        Each sample is created by:
            1. Randomly selecting rules to trigger
            2. Generating proof trees for those rules (using KGenerator)
            3. Extracting all atoms from proof trees (using shared utility)
            4. Converting atoms to KG format (using shared utility)
            5. Adding negative samples using local corruption (work in progress)

        Args:
            n_samples (int):        Number of samples to generate.
            min_individuals (int):  Min individuals per sample.
            max_individuals (int):  Max individuals per sample.
            min_rules (int):        Min rules per sample.
            max_rules (int):        Max rules per sample.
            sample_type (str):      "TRAIN" or "TEST" (for logging).

        Returns:
            List[KnowledgeGraph]: Generated samples.
        """
        samples = []
        failed_attempts = 0
        max_failed_attempts = n_samples * 2  # Safety limit

        while len(samples) < n_samples and failed_attempts < max_failed_attempts:
            sample = self._generate_one_sample(
                min_individuals=min_individuals,
                max_individuals=max_individuals,
                min_rules=min_rules,
                max_rules=max_rules,
            )

            if sample is not None:
                samples.append(sample)
                if len(samples) % 100 == 0 or len(samples) == n_samples:
                    print(
                        f"  [{sample_type}] Generated {len(samples)}/{n_samples} samples"
                    )
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
    ) -> Optional[KnowledgeGraph]:
        """
        Generates one complete, independent knowledge graph sample.

            1. Randomly select K rules as starting points (K \in [min_rules, max_rules])
            2. For each rule, generate M proof trees (M \in [1, 5])
            3. Extract ALL atoms from all proof trees (using shared utility)
            4. Convert atoms to KG format (using shared utility)
            5. Add negative samples using local CWA corruption
            6. Verify sample meets size constraints

        Args:
            min_individuals (int):  Minimum individuals required.
            max_individuals (int):  Maximum individuals allowed.
            min_rules (int):        Minimum rules to trigger.
            max_rules (int):        Maximum rules to trigger.

        Returns:
            Optional[KnowledgeGraph]: Generated sample, or None if generation failed.
        """
        if not self.rules:
            if self.verbose:
                print("Warning: No rules available for generation")
            return None

        # Randomly select starting rules
        n_rules = random.randint(min_rules, min(max_rules, len(self.rules)))
        selected_rules = random.sample(self.rules, n_rules)

        all_atoms: Set[Atom] = set()

        # Generate proofs and extract atoms
        for rule in selected_rules:
            try:
                # REFACTORED: Use KGenerator's generate_proofs_for_rule method
                proofs = self.generator.generate_proofs_for_rule(
                    rule_name=rule.name,
                    max_proofs=5,  # Randomly select up to 5 proofs per rule
                )

                if not proofs:
                    continue

                # Randomly select some proofs (for variety)
                n_proofs = random.randint(1, len(proofs))
                selected_proofs = random.sample(proofs, n_proofs)

                for proof in selected_proofs:
                    all_atoms.update(extract_all_atoms_from_proof(proof))

            except Exception as e:
                if self.verbose:
                    print(
                        f"Warning: Failed to generate proofs for rule {rule.name}: {e}"
                    )
                continue

        # Check if we have any atoms
        if not all_atoms:
            return None

        # Convert atoms to knowledge graph
        kg = atoms_to_knowledge_graph(
            atoms=all_atoms,
            schema_classes=self.schema_classes,
            schema_relations=self.schema_relations,
            schema_attributes=self.schema_attributes,
        )

        # Check size constraints
        n_individuals = len(kg.individuals)
        if n_individuals < min_individuals or n_individuals > max_individuals:
            return None

        # Add negative samples
        kg = self._add_negative_samples(kg)

        return kg

    def _add_negative_samples(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """
        Adds negative samples to the knowledge graph using local CWA.
        Note that we only generate negatives for positive triples.
        There are no negatives for memberships or attribute triples.

        (from RRN paper, Appendix D), but work in progress:
        "We generated exactly one negative inference for each positive
        inference that exists in the data by corrupting each of these
        positive inferences exactly once."

        For each positive triple:
            1. Randomly corrupt either subject OR object
            2. Verify the corrupted triple doesn't create inconsistency
            3. Add as negative triple

        EXAMPLE:
            Positive: parent(Ind_0, Ind_1)
            Negative: parent(Ind_0, Ind_3)   [corrupted object]
                   OR parent(Ind_2, Ind_1)   [corrupted subject]

        Args:
            kg (KnowledgeGraph): KG with only positive triples.

        Returns:
            KnowledgeGraph: KG with balanced positive/negative triples.
        """
        positive_triples = []
        negative_triples = []

        for triple in kg.triples:
            if triple.positive:
                positive_triples.append(triple)
            else:
                negative_triples.append(triple)

        # Generate one negative for each positive
        for pos_triple in positive_triples:
            max_attempts = 10  # Limit attempts to avoid infinite loops

            for attempt in range(max_attempts):
                neg_triple = self._corrupt_triple(pos_triple, kg)

                # Verify consistency: negative shouldn't conflict with positives
                if not self._creates_inconsistency(neg_triple, kg):
                    negative_triples.append(neg_triple)
                    break

        # Add negatives to KG
        kg.triples.extend(negative_triples)

        if self.verbose and len(negative_triples) < len(positive_triples):
            print(
                f"Warning: Generated {len(negative_triples)}/{len(positive_triples)} negatives"
            )

        classes = {cls.name for cls in self.schema_classes.values()}

        # TODO add way more sophisticated negative sampling!

        return kg

    def _corrupt_triple(self, triple: Triple, kg: KnowledgeGraph) -> Triple:
        """
        Creates a negative triple by corrupting subject or object.

        Randomly chooses to corrupt either:
            - Subject: Replace with random different individual
            - Object: Replace with random different individual

        Args:
            triple (Triple): Positive triple to corrupt.
            kg (KnowledgeGraph): Current knowledge graph (for individual pool).

        Returns:
            Triple: Corrupted negative triple.
        """
        if random.random() < 0.5:
            # Corrupt subject
            candidates = [i for i in kg.individuals if i != triple.subject]
            if not candidates:
                # Fallback: corrupt object instead
                candidates = [i for i in kg.individuals if i != triple.object]
                new_obj = random.choice(candidates)
                return Triple(
                    triple.subject, triple.predicate, new_obj, positive=False, proofs=[]
                )

            new_subj = random.choice(candidates)
            return Triple(
                new_subj, triple.predicate, triple.object, positive=False, proofs=[]
            )
        else:
            # Corrupt object
            candidates = [i for i in kg.individuals if i != triple.object]
            if not candidates:
                # Fallback: corrupt subject instead
                candidates = [i for i in kg.individuals if i != triple.subject]
                new_subj = random.choice(candidates)
                return Triple(
                    new_subj, triple.predicate, triple.object, positive=False, proofs=[]
                )

            new_obj = random.choice(candidates)
            return Triple(
                triple.subject, triple.predicate, new_obj, positive=False, proofs=[]
            )

    def _creates_inconsistency(self, neg_triple: Triple, kg: KnowledgeGraph) -> bool:
        """
        Checks if a negative triple would create inconsistency.

        A negative triple is inconsistent if its positive version
        exists in the knowledge graph.

        Args:
            neg_triple (Triple): Negative triple to check.
            kg (KnowledgeGraph): Current knowledge graph.

        Returns:
            bool: True if inconsistent, False otherwise.
        """
        # Check if positive version exists
        for triple in kg.triples:
            if (
                triple.positive
                and triple.subject.name == neg_triple.subject.name
                and triple.predicate.name == neg_triple.predicate.name
                and triple.object.name == neg_triple.object.name
            ):
                return True

        return False

    def _print_dataset_summary(
        self,
        train_samples: List[KnowledgeGraph],
        test_samples: List[KnowledgeGraph],
    ) -> None:
        """
        Prints summary statistics for generated datasets.

        Args:
            train_samples (List[KG]): Training samples.
            test_samples (List[KG]): Test samples.
        """

        def compute_stats(samples: List[KnowledgeGraph]) -> Dict:
            """Compute statistics for a list of samples."""
            if not samples:
                return {}

            stats = {
                "n_samples": len(samples),
                "avg_individuals": sum(len(s.individuals) for s in samples)
                / len(samples),
                "avg_triples": sum(len(s.triples) for s in samples) / len(samples),
                "avg_memberships": sum(len(s.memberships) for s in samples)
                / len(samples),
                "avg_pos_triples": sum(
                    len([t for t in s.triples if t.positive]) for s in samples
                )
                / len(samples),
                "avg_neg_triples": sum(
                    len([t for t in s.triples if not t.positive]) for s in samples
                )
                / len(samples),
                "avg_neg_memberships": sum(
                    len([m for m in s.memberships if not m.is_member]) for s in samples
                )
                / len(samples),
                "avg_pos_memberships": sum(
                    len([m for m in s.memberships if m.is_member]) for s in samples
                )
                / len(samples),
            }
            return stats

        train_stats = compute_stats(train_samples)
        test_stats = compute_stats(test_samples)

        print(f"\n{'=' * 80}")
        print("DATASET GENERATION COMPLETE")
        print(f"{'=' * 80}")

        print("\nTRAINING SET:")
        print(f"  Samples:              {train_stats.get('n_samples', 0)}")
        print(f"  Avg individuals:      {train_stats.get('avg_individuals', 0):.1f}")
        print(f"  Avg total triples:    {train_stats.get('avg_triples', 0):.1f}")
        print(f"    - Positive:         {train_stats.get('avg_pos_triples', 0):.1f}")
        print(f"    - Negative:         {train_stats.get('avg_neg_triples', 0):.1f}")
        print(f"  Avg memberships:      {train_stats.get('avg_memberships', 0):.1f}")
        print(
            f"    - Positive:         {train_stats.get('avg_pos_memberships', 0):.1f}"
        )
        print(
            f"    - Negative:         {train_stats.get('avg_neg_memberships', 0):.1f}"
        )

        print("\nTEST SET:")
        print(f"  Samples:              {test_stats.get('n_samples', 0)}")
        print(f"  Avg individuals:      {test_stats.get('avg_individuals', 0):.1f}")
        print(f"  Avg total triples:    {test_stats.get('avg_triples', 0):.1f}")
        print(f"    - Positive:         {test_stats.get('avg_pos_triples', 0):.1f}")
        print(f"    - Negative:         {test_stats.get('avg_neg_triples', 0):.1f}")
        print(f"  Avg memberships:      {test_stats.get('avg_memberships', 0):.1f}")
        print(f"    - Positive:         {test_stats.get('avg_pos_memberships', 0):.1f}")
        print(f"    - Negative:         {test_stats.get('avg_neg_memberships', 0):.1f}")

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
    Saves a dataset (list of KG samples) to CSV files.

    Each sample is saved as a separate CSV file with format:
        {output_dir}/{prefix}_{index}.csv

    Each row represents one fact with columns:
        subject, predicate, object, label, fact_type

    Where:
        - subject: Individual name (e.g., "Ind_0")
        - predicate: Relation/Attribute name or "rdf:type"
        - object: Individual name, Class name, or literal value
        - label: "1" for positive, "0" for negative
        - fact_type: "triple", "membership", or "attribute"

    Args:
        samples (List[KnowledgeGraph]): List of KG samples to save.
        output_dir (str): Directory to save CSV files.
        prefix (str): Prefix for file names.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(samples)} samples to {output_dir}/")

    for idx, kg in enumerate(samples):
        file_path = output_path / f"{prefix}_{idx:05d}.csv"
        kg.to_csv(str(file_path))

        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            print(f"  Saved {idx + 1}/{len(samples)} samples")

    print(f"Dataset saved successfully to {output_dir}/")


def load_dataset_from_csv(
    input_dir: str,
    prefix: str = "sample",
    n_samples: Optional[int] = None,
) -> List[KnowledgeGraph]:
    """
    Loads a dataset from CSV files.

    Reads all CSV files matching pattern:
        {input_dir}/{prefix}_*.csv

    Args:
        input_dir (str): Directory containing CSV files.
        prefix (str): Prefix of file names to load.
        n_samples (Optional[int]): Max number of samples to load (None = all).

    Returns:
        List[KnowledgeGraph]: Loaded KG samples.
    """
    input_path = Path(input_dir)

    # Find all matching CSV files
    pattern = f"{prefix}_*.csv"
    csv_files = sorted(input_path.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {input_dir}/{pattern}")

    # Limit to n_samples if specified
    if n_samples is not None:
        csv_files = csv_files[:n_samples]

    print(f"Loading {len(csv_files)} samples from {input_dir}/")

    samples = []
    for idx, file_path in enumerate(csv_files):
        kg = KnowledgeGraph.from_csv(str(file_path))
        samples.append(kg)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(csv_files):
            print(f"  Loaded {idx + 1}/{len(csv_files)} samples")

    print(f"Dataset loaded successfully from {input_dir}/")
    return samples


# ============================================================================ #
#                              MAIN ENTRY POINT                                #
# ============================================================================ #


def main():
    """
    Main entry point for dataset generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate RRN training/testing datasets from an ontology with constraint checking"
    )
    parser.add_argument(
        "--ontology",
        type=str,
        required=True,
        help="Path to the ontology file (.ttl format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/out/",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=5,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=2,
        help="Number of test samples to generate",
    )
    parser.add_argument(
        "--min-individuals",
        type=int,
        default=5,
        help="Minimum individuals per sample",
    )
    parser.add_argument(
        "--max-individuals",
        type=int,
        default=30,
        help="Maximum individuals per sample",
    )
    parser.add_argument(
        "--max-recursion",
        type=int,
        default=6,
        help="Maximum recursion depth for rules",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Initialize generator
    generator = KGEDatasetGenerator(
        ontology_file=args.ontology,
        max_recursion=args.max_recursion,
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

    print("\n✓ Dataset generation complete!")
    print(f"  Training samples: {args.output}/train/")
    print(f"  Test samples: {args.output}/test/")


if __name__ == "__main__":
    main()
