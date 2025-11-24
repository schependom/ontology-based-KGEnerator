"""
DESCRIPTION

    Advanced negative sampling strategies for Knowledge Graph data generation.

    Strategies:
    1. Random corruption (baseline)
    2. Constrained corruption (respects domain/range)
    3. Proof-based corruption (corrupts base facts to falsify goals)
    4. Type-aware corruption (respects class memberships)

AUTHOR

    Vincent Van Schependom
"""

import os
import random
from typing import List, Set, Dict, Optional
from data_structures import *
from collections import defaultdict


class NegativeSampler:
    """
    Handles generation of negative samples using multiple strategies.
    """

    def __init__(
        self,
        schema_classes: Dict[str, Class],
        schema_relations: Dict[str, Relation],
        domains: Dict[str, Set[str]] = None,
        ranges: Dict[str, Set[str]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the negative sampler.

        Args:
            schema_classes: Dict of class name -> Class object
            schema_relations: Dict of relation name -> Relation object
            domains: Dict of relation name -> set of domain class names
            ranges: Dict of relation name -> set of range class names
            verbose: Enable debug output
        """
        self.schema_classes = schema_classes
        self.schema_relations = schema_relations
        self.domains = domains or {}
        self.ranges = ranges or {}
        self.verbose = verbose
        
        # Optimization: Sets for fast lookup of existing facts
        self.existing_triples: Set[str] = set()
        self.existing_memberships: Set[str] = set()
        
        # Track strategy usage
        self.strategy_usage: Dict[str, int] = defaultdict(int)

    def _index_existing_facts(self, kg: KnowledgeGraph):
        """Build sets of existing facts for O(1) lookup."""
        self.existing_triples.clear()
        self.existing_memberships.clear()
        
        for t in kg.triples:
            if t.positive:
                # Key: (subject, predicate, object)
                key = f"{t.subject.name}|{t.predicate.name}|{t.object.name}"
                self.existing_triples.add(key)
                
        for m in kg.memberships:
            if m.is_member:
                # Key: (individual, class)
                key = f"{m.individual.name}|{m.cls.name}"
                self.existing_memberships.add(key)

    def add_negative_samples(
        self,
        kg: KnowledgeGraph,
        strategy: str = "constrained",
        ratio: float = 1.0,
        corrupt_base_facts: bool = False,
        export_proofs: bool = False,
        output_dir: str = None,
    ) -> KnowledgeGraph:
        """
        Add negative samples to a knowledge graph.

        Args:
            kg: Knowledge graph to add negatives to
            strategy: Negative sampling strategy
                - "random": Random corruption
                - "constrained": Respects domain/range constraints
                - "proof_based": Corrupts base facts in proof trees
                - "type_aware": Considers class memberships
                - "mixed": Randomly selects one of the above for each sample
            ratio: Ratio of negative to positive samples (1.0 = balanced)
            corrupt_base_facts: If True, also corrupt base facts in proofs
            export_proofs: Whether to export visualizations of corrupted proofs
            output_dir: Directory to save visualizations

        Returns:
            Knowledge graph with negative samples added
        """
        # Index existing facts for fast lookup
        self._index_existing_facts(kg)
        
        if strategy != "mixed":
            # For single strategies, we can just count the number of negatives added
            # But we need to know how many were actually added. 
            # The methods return the KG, so we can check the difference in length.
            initial_len = len(kg.triples)
        
        if strategy == "random":
            kg = self._random_corruption(kg, ratio)
            self.strategy_usage["random"] += len(kg.triples) - initial_len
            return kg
        elif strategy == "constrained":
            kg = self._constrained_corruption(kg, ratio)
            self.strategy_usage["constrained"] += len(kg.triples) - initial_len
            return kg
        elif strategy == "proof_based":
            kg = self._proof_based_corruption(
                kg, ratio, corrupt_base_facts, export_proofs, output_dir
            )
            self.strategy_usage["proof_based"] += len(kg.triples) - initial_len
            return kg
        elif strategy == "type_aware":
            kg = self._type_aware_corruption(kg, ratio)
            self.strategy_usage["type_aware"] += len(kg.triples) - initial_len
            return kg
        elif strategy == "mixed":
            return self._mixed_corruption(
                kg, ratio, corrupt_base_facts, export_proofs, output_dir
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _random_corruption(self, kg: KnowledgeGraph, ratio: float) -> KnowledgeGraph:
        """
        Strategy 1: Random corruption (baseline).

        For each positive triple, randomly corrupt subject OR object with any individual.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []

        for _ in range(n_negatives):
            if not positive_triples:
                break

            pos_triple = random.choice(positive_triples)
            neg_triple = self._corrupt_triple_random(pos_triple, kg.individuals)

            if neg_triple and not self._is_positive_fact(neg_triple, kg):
                negative_triples.append(neg_triple)

        kg.triples.extend(negative_triples)

        # Also add negative memberships
        positive_memberships = [m for m in kg.memberships if m.is_member]
        n_neg_memberships = int(len(positive_memberships) * ratio)
        negative_memberships = []

        for _ in range(n_neg_memberships):
            if not positive_memberships:
                break

            pos_mem = random.choice(positive_memberships)
            neg_mem = self._corrupt_membership_random(
                pos_mem, list(self.schema_classes.values())
            )

            if neg_mem and not self._is_positive_membership(neg_mem, kg):
                negative_memberships.append(neg_mem)

        kg.memberships.extend(negative_memberships)

        return kg

    def _constrained_corruption(
        self, kg: KnowledgeGraph, ratio: float
    ) -> KnowledgeGraph:
        """
        Strategy 2: Constrained corruption.

        Respects domain/range constraints when corrupting.
        Only substitutes individuals that satisfy type constraints.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []

        # Build individual -> classes mapping
        ind_classes = self._build_individual_classes_map(kg)

        for _ in range(n_negatives):
            if not positive_triples:
                break

            pos_triple = random.choice(positive_triples)

            # Get valid candidates based on domain/range
            if random.random() < 0.5:
                # Corrupt subject (check domain)
                candidates = self._get_domain_candidates(
                    pos_triple.predicate, kg.individuals, ind_classes
                )
                candidates = [c for c in candidates if c != pos_triple.subject]
                if candidates:
                    new_subj = random.choice(candidates)
                    neg_triple = Triple(
                        new_subj,
                        pos_triple.predicate,
                        pos_triple.object,
                        positive=False,
                        proofs=[],
                    )
                else:
                    continue
            else:
                # Corrupt object (check range)
                candidates = self._get_range_candidates(
                    pos_triple.predicate, kg.individuals, ind_classes
                )
                candidates = [c for c in candidates if c != pos_triple.object]
                if candidates:
                    new_obj = random.choice(candidates)
                    neg_triple = Triple(
                        pos_triple.subject,
                        pos_triple.predicate,
                        new_obj,
                        positive=False,
                        proofs=[],
                    )
                else:
                    continue

            if not self._is_positive_fact(neg_triple, kg):
                negative_triples.append(neg_triple)

        kg.triples.extend(negative_triples)

        # Constrained membership negatives
        positive_memberships = [m for m in kg.memberships if m.is_member]
        n_neg_memberships = int(len(positive_memberships) * ratio)
        negative_memberships = []

        for _ in range(n_neg_memberships):
            if not positive_memberships:
                break

            pos_mem = random.choice(positive_memberships)

            # Get classes the individual is NOT in
            current_classes = {
                m.cls.name
                for m in kg.memberships
                if m.individual == pos_mem.individual and m.is_member
            }

            candidate_classes = [
                c for c in self.schema_classes.values() if c.name not in current_classes
            ]

            if candidate_classes:
                neg_cls = random.choice(candidate_classes)
                neg_mem = Membership(
                    pos_mem.individual, neg_cls, is_member=False, proofs=[]
                )
                negative_memberships.append(neg_mem)

        kg.memberships.extend(negative_memberships)

        return kg

    def _proof_based_corruption(
        self,
        kg: KnowledgeGraph,
        ratio: float,
        corrupt_base_facts: bool,
        export_proofs: bool = False,
        output_dir: str = None,
    ) -> KnowledgeGraph:
        """
        Strategy 3: Proof-based corruption.

        Corrupts facts in proof trees to create negatives that would falsify inferences.
        This creates harder negatives that test reasoning capabilities.

        Example:
            If hasGrandparent(A, C) is inferred from hasParent(A, B) ∧ hasParent(B, C),
            then corrupting hasParent(A, B) to hasParent(A, D) falsifies the inference.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []

        # Collect triples with proofs
        triples_with_proofs = [(t, t.proofs) for t in positive_triples if t.proofs]

        if not triples_with_proofs:
            # Fallback to random if no proofs available
            return self._random_corruption(kg, ratio)

        # Limit exported visualizations to prevent freezing
        exported_count = 0
        MAX_EXPORTS = 5

        attempts = 0
        max_attempts = n_negatives * 10  # Safety limit to prevent infinite loops

        while len(negative_triples) < n_negatives and attempts < max_attempts:
            attempts += 1

            if not triples_with_proofs:
                break

            # Pick a triple with proofs
            pos_triple, proofs = random.choice(triples_with_proofs)

            if not proofs:
                continue

            # Pick a random proof
            proof = random.choice(proofs)

            # Get base facts from this proof
            base_facts = proof.get_base_facts()

            # Filter out RDF.type facts if we are in corrupt_base_facts mode
            if corrupt_base_facts:
                base_facts = {bf for bf in base_facts if bf.predicate != RDF.type}

            neg_triple = None

            if not base_facts or not corrupt_base_facts:
                # Corrupt the goal instead
                neg_triple = self._corrupt_triple_random(pos_triple, kg.individuals)
            else:
                # Corrupt a base fact from the proof
                # This creates a negative that would break the inference chain
                base_fact = random.choice(list(base_facts))

                # Find corresponding triple in KG
                matching_triples = [
                    t
                    for t in kg.triples
                    if (
                        t.subject.name == base_fact.subject.name
                        and t.predicate.name == base_fact.predicate.name
                        and t.object.name == base_fact.object.name
                        and t.positive
                    )
                ]

                if matching_triples:
                    base_triple = matching_triples[0]
                    neg_triple = self._corrupt_triple_random(
                        base_triple, kg.individuals
                    )

                    # If we have a negative triple and we want to export proofs
                    if (
                        neg_triple
                        # and export_proofs
                        and output_dir
                        and exported_count < MAX_EXPORTS
                    ):
                        # Create atom from negative triple
                        new_atom = Atom(
                            predicate=neg_triple.predicate,
                            subject=neg_triple.subject,
                            object=neg_triple.object,
                        )

                        # Create corrupted proof
                        corrupted_proof = proof.corrupt_leaf(base_fact, new_atom)

                        # Save visualization
                        filename = f"corrupted_proof_{len(negative_triples)}_{pos_triple.subject.name}_{pos_triple.predicate.name}_{pos_triple.object.name}"
                        full_path = os.path.join(output_dir, filename)
                        corrupted_proof.save_visualization(full_path, format="png")
                        exported_count += 1
                        if self.verbose:
                            print(f"Exported corrupted proof visualization: {filename}")

            if neg_triple and not self._is_positive_fact(neg_triple, kg):
                negative_triples.append(neg_triple)

        kg.triples.extend(negative_triples)
        return kg

    def _type_aware_corruption(
        self, kg: KnowledgeGraph, ratio: float
    ) -> KnowledgeGraph:
        """
        Strategy 4: Type-aware corruption.

        Only corrupts with individuals of the same types to create semantically valid
        but factually incorrect negatives.

        Example:
            If hasParent(Person1, Person2), only corrupt with other Persons.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []

        # Build individual -> classes mapping
        ind_classes = self._build_individual_classes_map(kg)

        # Group individuals by their class sets
        class_groups = defaultdict(list)
        for ind in kg.individuals:
            classes = frozenset(ind_classes.get(ind, set()))
            class_groups[classes].append(ind)

        for _ in range(n_negatives):
            if not positive_triples:
                break

            pos_triple = random.choice(positive_triples)

            if random.random() < 0.5:
                # Corrupt subject with same-type individual
                subj_classes = frozenset(ind_classes.get(pos_triple.subject, set()))
                candidates = [
                    c for c in class_groups[subj_classes] if c != pos_triple.subject
                ]

                if candidates:
                    new_subj = random.choice(candidates)
                    neg_triple = Triple(
                        new_subj,
                        pos_triple.predicate,
                        pos_triple.object,
                        positive=False,
                        proofs=[],
                    )
                else:
                    continue
            else:
                # Corrupt object with same-type individual
                obj_classes = frozenset(ind_classes.get(pos_triple.object, set()))
                candidates = [
                    c for c in class_groups[obj_classes] if c != pos_triple.object
                ]

                if candidates:
                    new_obj = random.choice(candidates)
                    neg_triple = Triple(
                        pos_triple.subject,
                        pos_triple.predicate,
                        new_obj,
                        positive=False,
                        proofs=[],
                    )
                else:
                    continue

            if not self._is_positive_fact(neg_triple, kg):
                negative_triples.append(neg_triple)

        kg.triples.extend(negative_triples)
        kg.triples.extend(negative_triples)
        return kg

    def _mixed_corruption(
        self,
        kg: KnowledgeGraph,
        ratio: float,
        corrupt_base_facts: bool,
        export_proofs: bool,
        output_dir: str,
    ) -> KnowledgeGraph:
        """
        Strategy 5: Mixed corruption.
        
        Randomly selects a strategy for each negative sample to create a diverse dataset.
        """
        positive_triples = [t for t in kg.triples if t.positive]
        n_negatives = int(len(positive_triples) * ratio)
        negative_triples = []
        
        # Pre-calculate data structures needed for different strategies
        ind_classes = self._build_individual_classes_map(kg)
        
        # Group individuals by their class sets (for type_aware)
        class_groups = defaultdict(list)
        for ind in kg.individuals:
            classes = frozenset(ind_classes.get(ind, set()))
            class_groups[classes].append(ind)
            
        triples_with_proofs = [(t, t.proofs) for t in positive_triples if t.proofs]
        
        strategies = ["random", "constrained", "type_aware"]
        if triples_with_proofs:
            strategies.append("proof_based")
            
        for _ in range(n_negatives):
            if not positive_triples:
                break
                
            # Pick a random strategy for this sample
            strategy = random.choice(strategies)
            
            neg_triple = None
            
            if strategy == "random":
                pos_triple = random.choice(positive_triples)
                neg_triple = self._corrupt_triple_random(pos_triple, kg.individuals)
                
            elif strategy == "constrained":
                pos_triple = random.choice(positive_triples)
                if random.random() < 0.5:
                    # Corrupt subject
                    candidates = self._get_domain_candidates(
                        pos_triple.predicate, kg.individuals, ind_classes
                    )
                    candidates = [c for c in candidates if c != pos_triple.subject]
                    if candidates:
                        new_subj = random.choice(candidates)
                        neg_triple = Triple(new_subj, pos_triple.predicate, pos_triple.object, positive=False, proofs=[])
                else:
                    # Corrupt object
                    candidates = self._get_range_candidates(
                        pos_triple.predicate, kg.individuals, ind_classes
                    )
                    candidates = [c for c in candidates if c != pos_triple.object]
                    if candidates:
                        new_obj = random.choice(candidates)
                        neg_triple = Triple(pos_triple.subject, pos_triple.predicate, new_obj, positive=False, proofs=[])
                        
            elif strategy == "type_aware":
                pos_triple = random.choice(positive_triples)
                if random.random() < 0.5:
                    # Corrupt subject
                    subj_classes = frozenset(ind_classes.get(pos_triple.subject, set()))
                    candidates = [c for c in class_groups[subj_classes] if c != pos_triple.subject]
                    if candidates:
                        new_subj = random.choice(candidates)
                        neg_triple = Triple(new_subj, pos_triple.predicate, pos_triple.object, positive=False, proofs=[])
                else:
                    # Corrupt object
                    obj_classes = frozenset(ind_classes.get(pos_triple.object, set()))
                    candidates = [c for c in class_groups[obj_classes] if c != pos_triple.object]
                    if candidates:
                        new_obj = random.choice(candidates)
                        neg_triple = Triple(pos_triple.subject, pos_triple.predicate, new_obj, positive=False, proofs=[])
                        
            elif strategy == "proof_based":
                # Simplified proof based logic for single sample
                if triples_with_proofs:
                    pos_triple, proofs = random.choice(triples_with_proofs)
                    if proofs:
                        proof = random.choice(proofs)
                        base_facts = proof.get_base_facts()
                        
                        if corrupt_base_facts:
                            base_facts = {bf for bf in base_facts if bf.predicate != RDF.type}
                            
                        if not base_facts or not corrupt_base_facts:
                            neg_triple = self._corrupt_triple_random(pos_triple, kg.individuals)
                        else:
                            base_fact = random.choice(list(base_facts))
                            # Find matching triple
                            matching_triples = [
                                t for t in kg.triples 
                                if t.positive and 
                                t.subject.name == base_fact.subject.name and 
                                t.predicate.name == base_fact.predicate.name and 
                                t.object.name == base_fact.object.name
                            ]
                            if matching_triples:
                                neg_triple = self._corrupt_triple_random(matching_triples[0], kg.individuals)

            if neg_triple and not self._is_positive_fact(neg_triple, kg):
                negative_triples.append(neg_triple)
                self.strategy_usage[strategy] += 1
                
        kg.triples.extend(negative_triples)
        
        # For memberships, we just use random corruption for now as it's less critical
        # or we could implement mixed logic there too, but let's stick to random for simplicity in mixed mode
        # unless we want to be very thorough. Let's reuse random corruption for memberships.
        positive_memberships = [m for m in kg.memberships if m.is_member]
        n_neg_memberships = int(len(positive_memberships) * ratio)
        negative_memberships = []
        
        for _ in range(n_neg_memberships):
            if not positive_memberships:
                break
            pos_mem = random.choice(positive_memberships)
            neg_mem = self._corrupt_membership_random(pos_mem, list(self.schema_classes.values()))
            if neg_mem and not self._is_positive_membership(neg_mem, kg):
                negative_memberships.append(neg_mem)
                
        kg.memberships.extend(negative_memberships)
        return kg

    # ==================== HELPER METHODS ==================== #

    def _corrupt_triple_random(
        self, triple: Triple, individuals: List[Individual]
    ) -> Optional[Triple]:
        """Randomly corrupt subject or object of a triple."""
        if not individuals:
            return None

        if random.random() < 0.5:
            # Corrupt subject
            candidates = [i for i in individuals if i != triple.subject]
            if not candidates:
                return None
            new_subj = random.choice(candidates)
            return Triple(
                new_subj, triple.predicate, triple.object, positive=False, proofs=[]
            )
        else:
            # Corrupt object
            candidates = [i for i in individuals if i != triple.object]
            if not candidates:
                return None
            new_obj = random.choice(candidates)
            return Triple(
                triple.subject, triple.predicate, new_obj, positive=False, proofs=[]
            )

    def _corrupt_membership_random(
        self, membership: Membership, classes: List[Class]
    ) -> Optional[Membership]:
        """Randomly corrupt class membership."""
        if not classes:
            return None
        candidates = [c for c in classes if c != membership.cls]
        if not candidates:
            return None
        new_cls = random.choice(candidates)
        return Membership(membership.individual, new_cls, is_member=False, proofs=[])

    def _build_individual_classes_map(
        self, kg: KnowledgeGraph
    ) -> Dict[Individual, Set[str]]:
        """Build mapping from individuals to their classes."""
        ind_classes = defaultdict(set)
        for mem in kg.memberships:
            if mem.is_member:
                ind_classes[mem.individual].add(mem.cls.name)
        return ind_classes

    def _get_domain_candidates(
        self,
        relation: Relation,
        individuals: List[Individual],
        ind_classes: Dict[Individual, Set[str]],
    ) -> List[Individual]:
        """Get individuals that satisfy domain constraints for a relation."""
        required_classes = self.domains.get(relation.name, set())

        if not required_classes:
            return individuals

        candidates = []
        for ind in individuals:
            ind_cls = ind_classes.get(ind, set())
            if not required_classes.isdisjoint(ind_cls):
                candidates.append(ind)

        return candidates if candidates else individuals

    def _get_range_candidates(
        self,
        relation: Relation,
        individuals: List[Individual],
        ind_classes: Dict[Individual, Set[str]],
    ) -> List[Individual]:
        """Get individuals that satisfy range constraints for a relation."""
        required_classes = self.ranges.get(relation.name, set())

        if not required_classes:
            return individuals

        candidates = []
        for ind in individuals:
            ind_cls = ind_classes.get(ind, set())
            if not required_classes.isdisjoint(ind_cls):
                candidates.append(ind)

        return candidates if candidates else individuals

    def _is_positive_fact(self, neg_triple: Triple, kg: KnowledgeGraph) -> bool:
        """Check if a negative triple conflicts with a positive fact."""
        # Optimized O(1) lookup
        key = f"{neg_triple.subject.name}|{neg_triple.predicate.name}|{neg_triple.object.name}"
        return key in self.existing_triples

    def _is_positive_membership(self, neg_mem: Membership, kg: KnowledgeGraph) -> bool:
        """Check if a negative membership conflicts with a positive one."""
        # Optimized O(1) lookup
        key = f"{neg_mem.individual.name}|{neg_mem.cls.name}"
        return key in self.existing_memberships
