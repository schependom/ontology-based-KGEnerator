"""
DESCRIPTION:

    Knowledge Graph Generator.

    Running this file as a script will execute the full data generation
    pipeline from ontology to complete knowledge graph. Note that the Knowledge
    Graph generated here is the FULL graph with ALL derivable facts, so may be
    extremely large depending on the ontology and rules.

    Use create_data.py to generate train/test splits.

AUTHOR

    Vincent Van Schependom
"""

from collections import defaultdict
import sys
import argparse
import traceback
import random
from typing import Dict, Set, List, Optional
from rdflib.namespace import RDF

# Custom imports
from data_structures import (
    KnowledgeGraph,
    Individual,
    Class,
    Relation,
    Attribute,
    Triple,
    Membership,
    AttributeTriple,
    Atom,
    Proof,
    Term,
)
from parser import OntologyParser
from chainer import BackwardChainer
from graph_visualizer import GraphVisualizer


# ============================================================================ #
#                         SHARED UTILITY FUNCTIONS                             #
# ============================================================================ #


def extract_proof_map(proof: Proof) -> Dict[Atom, List[Proof]]:
    """
    Recursively extracts atoms and maps them to the proofs that derive them.

    Unlike 'extract_all_atoms_from_proof', this preserves the Proof objects,
    allowing us to determine if a fact was derived (Rule) or is base (None).
    """
    # map: Atom -> List[Proof]
    proof_map = defaultdict(list)

    # 1. Map the goal of this specific proof node
    proof_map[proof.goal].append(proof)

    # 2. Recursively collect from sub-proofs
    for sub_proof in proof.sub_proofs:
        sub_map = extract_proof_map(sub_proof)
        for atom, proofs in sub_map.items():
            proof_map[atom].extend(proofs)

    return proof_map


def extract_all_atoms_from_proof(proof: Proof) -> Set[Atom]:
    """
    Recursively extracts ALL atoms from a proof tree (DFS traversal).

    This includes:
    - The goal atom (conclusion)
    - All atoms from sub-proofs (premises)
    - Atoms at all levels (base facts + intermediate inferences)

    RUNNING EXAMPLE:
    ---------------
    Given proof tree:
        grandparent(Ind_0, Ind_2)   [GOAL]
        ├─ parent(Ind_0, Ind_1)     [premise 1]
        │  └─ child(Ind_1, Ind_0)   [base fact]
        └─ parent(Ind_1, Ind_2)     [premise 2]
           └─ child(Ind_2, Ind_1)   [base fact]

    Returns: {
        grandparent(Ind_0, Ind_2),  # inferred
        parent(Ind_0, Ind_1),       # inferred
        child(Ind_1, Ind_0),        # base
        parent(Ind_1, Ind_2),       # inferred
        child(Ind_2, Ind_1),        # base
    }

    Args:
        proof (Proof): The proof tree to extract atoms from.

    Returns:
        Set[Atom]: All ground atoms in the proof tree.
    """
    atoms: Set[Atom] = set()

    # Add the goal of this proof
    atoms.add(proof.goal)

    # Recursively add atoms from all sub-proofs
    for sub_proof in proof.sub_proofs:
        atoms.update(extract_all_atoms_from_proof(sub_proof))

    return atoms


def atoms_to_knowledge_graph(
    atoms: Set[Atom],
    schema_classes: Dict[str, Class],
    schema_relations: Dict[str, Relation],
    schema_attributes: Dict[str, Attribute],
    proof_map: Optional[Dict[Atom, List[Proof]]],
) -> KnowledgeGraph:
    """
    Converts a set of ground atoms into a KnowledgeGraph.

    This organizes atoms into:
        - Memberships:      (Individual, rdf:type, Class)
        - Triples:          (Individual, Relation, Individual)
        - AttributeTriples: (Individual, Attribute, LiteralValue)

    All facts are positive at this stage. Negatives should be added separately.

    Args:
        atoms (Set[Atom]):                          Set of ground atoms to convert.
        schema_classes (Dict[str, Class]):          Schema classes from parser.
        schema_relations (Dict[str, Relation]):     Schema relations from parser.
        schema_attributes (Dict[str, Attribute]):   Schema attributes from parser.

    Returns:
        KnowledgeGraph: Organized knowledge graph structure.
    """
    # Storage for KG components
    individuals: Dict[str, Individual] = {}
    triples: Dict[tuple, Triple] = {}
    memberships: Dict[tuple, Membership] = {}
    attr_triples: Dict[tuple, AttributeTriple] = {}

    def register_individual(term) -> None:
        """Helper to register an individual if not seen before."""
        if isinstance(term, Individual):
            if term.name not in individuals:
                individuals[term.name] = term

    # Convert each atom to appropriate KG structure
    for atom in atoms:
        s, p, o = atom.subject, atom.predicate, atom.object

        # Get proofs for this atom if available
        current_proofs = []
        if proof_map and atom in proof_map:
            current_proofs = proof_map[atom]

        # 1. MEMBERSHIPS
        if p == RDF.type and isinstance(o, Class):
            register_individual(s)
            key = (s.name, o.name)
            if key not in memberships:
                memberships[key] = Membership(s, o, True, proofs=[])

            # Attach proofs
            if current_proofs:
                memberships[key].proofs.extend(current_proofs)

        # 2. RELATIONAL TRIPLES
        elif isinstance(o, Individual) and isinstance(p, Relation):
            register_individual(s)
            register_individual(o)
            key = (s.name, p.name, o.name)
            if key not in triples:
                triples[key] = Triple(s, p, o, True, proofs=[])

            # Attach proofs
            if current_proofs:
                triples[key].proofs.extend(current_proofs)

        # 3. ATTRIBUTES
        elif isinstance(p, Attribute):
            register_individual(s)
            key = (s.name, p.name, o)
            if key not in attr_triples:
                attr_triples[key] = AttributeTriple(s, p, o, proofs=[])

            # Attach proofs
            if current_proofs:
                attr_triples[key].proofs.extend(current_proofs)

    # Create knowledge graph
    return KnowledgeGraph(
        attributes=list(schema_attributes.values()),
        classes=list(schema_classes.values()),
        relations=list(schema_relations.values()),
        individuals=list(individuals.values()),
        triples=list(triples.values()),
        memberships=list(memberships.values()),
        attribute_triples=list(attr_triples.values()),
    )


# ============================================================================ #
#                           CORE GENERATOR CLASS                               #
# ============================================================================ #


class KGenerator:
    """
    Knowledge Graph data Generator (KGenerator).

    Orchestrates the full data generation pipeline from ontology to
    complete 1 knowledge graph with all derivable facts.

    The generator uses backward chaining to explore all possible proof
    paths in the ontology, generating individuals and facts along the way,
    while ensuring all constraints are satisfied.
    """

    def __init__(
        self,
        ontology_file: str,
        max_recursion: int,
        global_max_depth: int,
        individual_pool_size: int,
        individual_reuse_prob: float,
        max_proofs_per_atom: int = None,
        neg_strategy: str = "random",
        verbose: bool = False,
        export_proof_visualizations=False,
    ):
        """
        Initializes the parser and chainer.

        Args:
            ontology_file (str):                Path to the .ttl ontology file.
            max_recursion (int):                The maximum depth for recursive rules.
                                                This prevents infinite recursion in rules like:
                                                    parent(X,Y) ∧ parent(Y,Z) → ancestor(X,Z)
            global_max_depth (int):             Hard limit on total proof tree depth.
            individual_pool_size (int):         Size of the individual pool for generation.
            individual_reuse_prob (float):      Probability of reusing existing individuals.
            max_proofs_per_atom (int):          Max number of proofs to generate for any single atom.
            neg_strategy (str):                 "random", "constrained", "proof_based", "type_aware"
            verbose (bool):                     Enable detailed logging.
            export_proof_visualizations (bool): Whether to export proof visualizations.
        """
        self.verbose = verbose
        self.neg_strategy = neg_strategy

        if self.verbose:
            print(f"Loading and parsing ontology from: {ontology_file}")

        # Parse the ontology to extract schema, rules, and constraints
        self.parser = OntologyParser(ontology_file)

        # Initialize the backward chainer with all rules AND constraints
        if self.verbose:
            print(
                f"Initializing backward chainer with {len(self.parser.constraints)} constraints..."
            )
        self.chainer = BackwardChainer(
            all_rules=self.parser.rules,
            constraints=self.parser.constraints,
            max_recursion_depth=max_recursion,
            individual_pool_size=individual_pool_size,
            individual_reuse_prob=individual_reuse_prob,
            global_max_depth=global_max_depth,
            max_proofs_per_atom=max_proofs_per_atom,
            verbose=verbose,
            export_proof_visualizations=export_proof_visualizations,
        )

        # Store schemas from the parser
        self.schema_classes = self.parser.classes
        self.schema_relations = self.parser.relations
        self.schema_attributes = self.parser.attributes

        # Initialize storage for generated data
        # We use dictionaries with unique keys for automatic deduplication
        self.individuals: Dict[str, Individual] = {}  # name -> Individual
        self.triples: Dict[tuple, Triple] = {}  # (subj, pred, obj) -> Triple
        self.memberships: Dict[tuple, Membership] = {}  # (ind, cls) -> Membership
        self.attr_triples: Dict[
            tuple, AttributeTriple
        ] = {}  # (subj, pred, val) -> AttributeTriple

        # Track processed proofs to avoid duplicate work
        # This is critical because proof trees form a DAG structure
        # Example: If proofs A→B and A→C both depend on D, we only process D once
        self.processed_proofs: Set[Proof] = set()

        # Statistics tracking
        self.stats = {
            "proofs_generated": 0,
            "proofs_rejected_by_constraints": 0,
            "proofs_accepted": 0,
        }

        # Negative sampling strategy
        self.neg_strategy = "random"

    def generate_proofs_for_rule(
        self,
        rule_name: str,
        max_proofs: Optional[int] = None,
    ) -> List[Proof]:
        """
        Generate proof trees for a specific rule.

        Example:

            rule_name = "owl_chain_hasParent_hasParent_hasGrandparent"

            This might generate proofs like:
                1. grandparent(Ind_0, Ind_2) ← parent(Ind_0, Ind_1) ∧ parent(Ind_1, Ind_2)
                2. grandparent(Ind_3, Ind_5) ← parent(Ind_3, Ind_4) ∧ parent(Ind_4, Ind_5)
            ...

        Args:
            rule_name (str):            Name of the rule to generate proofs for.
            max_proofs (Optional[int]): Maximum number of proofs to generate.
                                        None = generate all possible proofs.

        Returns:
            List[Proof]: List of valid proof trees (constraint-checked).
        """
        if rule_name not in self.chainer.all_rules:
            if self.verbose:
                print(f"Warning: Rule '{rule_name}' not found.")
            return []

        proofs = []
        proof_generator = self.chainer.generate_proof_trees(rule_name)

        try:
            for i, proof in enumerate(proof_generator):
                if max_proofs is not None and i >= max_proofs:
                    break
                proofs.append(proof)
                self.stats["proofs_accepted"] += 1

        except Exception as e:
            if self.verbose:
                print(f"Error generating proofs for {rule_name}: {e}")
                traceback.print_exc()

        return proofs

    def _register_individual(self, term: Term) -> None:
        """
        Adds an Individual to our store if it's not already present.

        This ensures each individual is only stored once, even if it appears
        in multiple facts.

        Args:
            term (Term): A term that should be an Individual.
        """
        if isinstance(term, Individual):
            if term.name not in self.individuals:
                self.individuals[term.name] = term
        else:
            if self.verbose:
                print(f"Warning: Expected Individual term, got: {term}")

    def _add_atom_and_proof(self, atom: Atom, proof: Proof) -> None:
        """
        Converts a ground Atom into a KG fact and adds it to our stores.

        If the fact already exists, appends this proof to its proof list.
        This allows tracking multiple derivation paths for the same fact.

        Args:
            atom (Atom):    The ground atom to add (must be fully ground).
            proof (Proof):  The proof object that derived this atom.
        """
        # Sanity check: goal of a proof must always be ground
        if not atom.is_ground():
            if self.verbose:
                print(f"Warning: Skipping non-ground atom: {atom}")
            return

        # Extract components of the atom
        s, p, o = atom.subject, atom.predicate, atom.object

        # ==================== CLASS MEMBERSHIPS ==================== #
        # Pattern: (Individual, rdf:type, Class)
        if p == RDF.type and isinstance(o, Class):
            self._register_individual(s)
            key = (s.name, o.name)
            if key not in self.memberships:
                # Create new membership fact
                self.memberships[key] = Membership(
                    individual=s, cls=o, is_member=True, proofs=[]
                )
            # Append this proof to the fact's proof list
            self.memberships[key].proofs.append(proof)

        # ==================== RELATIONAL TRIPLES ==================== #
        # Pattern: (Individual, Relation, Individual)
        elif isinstance(o, Individual) and isinstance(p, Relation):
            self._register_individual(s)
            self._register_individual(o)
            key = (s.name, p.name, o.name)
            if key not in self.triples:
                # Create new relational triple
                self.triples[key] = Triple(
                    subject=s, predicate=p, object=o, positive=True, proofs=[]
                )
            # Append this proof to the fact's proof list
            self.triples[key].proofs.append(proof)

        # ==================== ATTRIBUTE TRIPLES ==================== #
        # Pattern: (Individual, Attribute, LiteralValue)
        elif isinstance(p, Attribute):
            self._register_individual(s)
            # Note: object (o) is a LiteralValue (string, int, etc.)
            key = (s.name, p.name, o)
            if key not in self.attr_triples:
                # Create new attribute triple
                self.attr_triples[key] = AttributeTriple(
                    subject=s, predicate=p, value=o, proofs=[]
                )
            # Append this proof to the fact's proof list
            self.attr_triples[key].proofs.append(proof)

        # Other atom patterns (e.g., variable predicates) are not handled

    def generate_data(self, add_negatives: bool = False) -> KnowledgeGraph:
        """
        Runs the full generation process.

        Args:
            add_negatives (bool): Whether to generate negative samples.

        Returns:
            KnowledgeGraph: The complete generated knowledge graph.
        """
        all_rules = self.parser.rules

        # Early exit if no rules
        if not all_rules:
            if self.verbose:
                print(
                    "Warning: No rules found in the ontology. No data will be generated."
                )
            return self.build_knowledge_graph()

        if self.verbose:
            print(f"Starting data generation from {len(all_rules)} rules...")

        # Iterate through all rules as starting points
        for i, rule in enumerate(all_rules):
            if self.verbose:
                print(f"\n[{i + 1}/{len(all_rules)}] Generating from rule: {rule.name}")

            # Generate all top-level proof trees starting from this rule
            # NOTE: The chainer now internally checks constraints before yielding proofs
            proof_generator = self.chainer.generate_proof_trees(rule.name)
            top_level_proofs = 0

            # Process each top-level proof (these are already validated)
            for top_level_proof in proof_generator:
                top_level_proofs += 1
                self.stats["proofs_accepted"] += 1

                # Traverse the entire proof tree (DAG) from the top
                # We use depth-first traversal with a stack
                stack = [top_level_proof]

                while stack:
                    current_proof = stack.pop()

                    # Skip if we've already processed this proof
                    if current_proof in self.processed_proofs:
                        continue

                    # Mark as processed
                    self.processed_proofs.add(current_proof)

                    # Add the goal of this proof to our KG
                    self._add_atom_and_proof(current_proof.goal, current_proof)

                    # Add all sub-proofs to the stack for processing
                    for sub_proof in current_proof.sub_proofs:
                        stack.append(sub_proof)

            if self.verbose:
                if top_level_proofs == 0:
                    print(
                        "  -> No valid proofs found (rule may not be a valid starting point)."
                    )
                else:
                    print(
                        f"  -> Found and processed {top_level_proofs} valid proof tree(s)."
                    )

        if self.verbose:
            print("\n--- Data generation complete. ---")
            print("Statistics:")
            print(f"  Total proofs accepted: {self.stats['proofs_accepted']}")

        # Build the initial KG
        kg = self.build_knowledge_graph()

        # Add negative samples if requested
        if add_negatives:
            if self.verbose:
                print("\nGenerating negative samples...")
            kg = self.add_negative_samples(kg)

        return kg

    def add_negative_samples(self, kg: KnowledgeGraph) -> KnowledgeGraph:
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
        # 1. RELATION NEGATIVES
        positive_triples = [t for t in kg.triples if t.positive]
        negative_triples = []

        for pos_triple in positive_triples:
            max_attempts = 5
            for _ in range(max_attempts):
                neg_triple = self._corrupt_triple(pos_triple, kg)
                if neg_triple and not self._creates_inconsistency(neg_triple, kg):
                    negative_triples.append(neg_triple)
                    break
        kg.triples.extend(negative_triples)

        # 2. CLASS MEMBERSHIP NEGATIVES
        # Generate one negative membership for each positive membership
        positive_memberships = [m for m in kg.memberships if m.is_member]
        negative_memberships = []

        all_classes = self.schema_classes.values()

        for pos_mem in positive_memberships:
            # Attempt to corrupt class (e.g., Person -> Building)
            # Logic: Pick a random class that the individual does NOT have
            current_classes = {
                m.cls.name
                for m in kg.memberships
                if m.individual == pos_mem.individual and m.is_member
            }

            candidates = [c for c in all_classes if c.name not in current_classes]
            if candidates:
                neg_cls = random.choice(candidates)
                # For a negative sample (False fact), we want facts that are NOT true.
                # Being disjoint makes it definitely false (a good negative).
                neg_mem = Membership(
                    individual=pos_mem.individual,
                    cls=neg_cls,
                    is_member=False,  # Explicitly negative
                    proofs=[],
                )
                negative_memberships.append(neg_mem)

        kg.memberships.extend(negative_memberships)

        # No attribute negatives yet (20/11/2026)

        return kg

    def _get_valid_candidates(
        self, relation: Relation, position: str, kg: KnowledgeGraph
    ) -> List[Individual]:
        """
        Get candidate individuals for corruption that satisfy domain/range constraints.

        Args:
            relation (Relation): The relation being corrupted.
            position (str): "subject" (check domain) or "object" (check range).
            kg (KnowledgeGraph): The knowledge graph containing individuals.

        Returns:
            List[Individual]: List of valid candidates.
        """
        if self.neg_strategy == "random":
            return kg.individuals

        # Filter by domain/range
        required_classes = set()
        if position == "subject":
            required_classes = self.parser.domains.get(relation.name, set())
        else:
            required_classes = self.parser.ranges.get(relation.name, set())

        if not required_classes:
            # No constraints, any individual is valid
            return kg.individuals

        valid_candidates = []
        for ind in kg.individuals:
            # Check if individual belongs to ANY of the required classes
            # (Union semantics for multiple domain/range statements)
            ind_classes = {m.cls.name for m in ind.classes if m.is_member}

            # If individual has no known classes, we might assume it's valid or invalid
            # Here we assume invalid if constraints exist but individual has no types
            if not ind_classes:
                continue

            # Check intersection
            if not required_classes.isdisjoint(ind_classes):
                valid_candidates.append(ind)

        # Fallback if no valid candidates found (to avoid empty list)
        if not valid_candidates:
            return kg.individuals

        return valid_candidates

    def _corrupt_triple(self, triple: Triple, kg: KnowledgeGraph) -> Optional[Triple]:
        """
        Creates a negative triple by corrupting subject or object.

        Randomly chooses to corrupt either:
            - Subject: Replace with random different individual
            - Object: Replace with random different individual

        Args:
            triple (Triple): Positive triple to corrupt.
            kg (KnowledgeGraph): Current knowledge graph (for individual pool).

        Returns:
            Optional[Triple]: Corrupted negative triple, or None if no candidates.
        """
        if random.random() < 0.5:
            # Corrupt subject
            candidates = self._get_valid_candidates(triple.predicate, "subject", kg)
            candidates = [i for i in candidates if i != triple.subject]

            if not candidates:
                # Fallback: corrupt object instead
                candidates = self._get_valid_candidates(triple.predicate, "object", kg)
                candidates = [i for i in candidates if i != triple.object]
                if not candidates:
                    return None
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
            candidates = self._get_valid_candidates(triple.predicate, "object", kg)
            candidates = [i for i in candidates if i != triple.object]

            if not candidates:
                # Fallback: corrupt subject instead
                candidates = self._get_valid_candidates(triple.predicate, "subject", kg)
                candidates = [i for i in candidates if i != triple.subject]
                if not candidates:
                    return None
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

    def build_knowledge_graph(self) -> KnowledgeGraph:
        """
        Assembles the final KnowledgeGraph object from all collected data.

        Returns:
            KnowledgeGraph: Complete knowledge graph with schema and generated data.
        """
        return KnowledgeGraph(
            # Schema elements from parser
            attributes=list(self.schema_attributes.values()),
            classes=list(self.schema_classes.values()),
            relations=list(self.schema_relations.values()),
            # Generated data from chainer
            individuals=list(self.individuals.values()),
            triples=list(self.triples.values()),
            memberships=list(self.memberships.values()),
            attribute_triples=list(self.attr_triples.values()),
        )


# ============================================================================ #
#                         COMMAND-LINE INTERFACE                               #
# ============================================================================ #


def main():
    """
    Main entry point for the script.
    """
    # ==================== PARSE ARGUMENTS ==================== #

    default_ontology_path = "data/toy.ttl"
    default_max_recursion = 3
    default_global_max_depth = 5
    default_max_proofs = 5

    parser = argparse.ArgumentParser(
        description="Ontology-based Knowledge Graph Data Generator with Constraint Checking"
    )
    parser.add_argument(
        "--ontology-path",
        type=str,
        default=default_ontology_path,
        help=f"Path to the ontology file (default: '{default_ontology_path}')",
    )
    parser.add_argument(
        "--max-recursion",
        type=int,
        default=default_max_recursion,
        help=f"Maximum depth for recursive rules (default: {default_max_recursion})",
    )
    parser.add_argument(
        "--global-max-depth",
        type=int,
        default=default_global_max_depth,
        help=f"Hard limit on total proof tree depth (default: {default_global_max_depth})",
    )
    parser.add_argument(
        "--max-proofs-per-atom",
        type=int,
        default=default_max_proofs,
        help="Max number of proofs to generate for any single atom (default: None)",
    )
    parser.add_argument(
        "--add-negatives",
        action="store_true",
        help="Generate negative samples (local CWA)",
    )
    parser.add_argument(
        "--individual-pool-size",
        type=int,
        default=50000,
        help="Size of the individual pool for generation (default: 50000)",
    )
    parser.add_argument(
        "--individual-reuse-prob",
        type=float,
        default=0.7,
        help="Probability of reusing existing individuals (default: 0.7)",
    )
    parser.add_argument(
        "--neg-strategy",
        type=str,
        choices=["random", "constrained", "proof_based", "type_aware"],
        default="random",
        help="Negative sampling strategy (default: 'random')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # ==================== RUN GENERATION PIPELINE ==================== #

    try:
        # Initialize Generator
        generator = KGenerator(
            args.ontology_path,
            max_recursion=args.max_recursion,
            global_max_depth=args.global_max_depth,
            individual_pool_size=args.individual_pool_size,
            individual_reuse_prob=args.individual_reuse_prob,
            max_proofs_per_atom=args.max_proofs_per_atom,
            neg_strategy=args.neg_strategy,
            verbose=args.verbose,
            export_proof_visualizations=False,
        )

        # Run Generation
        kg = generator.generate_data(add_negatives=args.add_negatives)

        # Print Summary
        print("\n--- Knowledge Graph Generation Summary ---")
        print(f"  Schema Classes:    {len(kg.classes)}")
        print(f"  Schema Relations:  {len(kg.relations)}")
        print(f"  Schema Attributes: {len(kg.attributes)}")
        print("------------------------------------------")
        print(f"  Generated Individuals: {len(kg.individuals)}")
        print(f"  Generated Triples:     {len(kg.triples)}")
        print(f"  Generated Memberships: {len(kg.memberships)}")
        print(f"  Generated AttrTriples: {len(kg.attribute_triples)}")
        print("------------------------------------------")

        # Print full knowledge graph
        kg.print()

        # Visualize
        print("\nVisualizing Knowledge Graph...")
        visualizer = GraphVisualizer("full-graphs")
        visualizer.visualize(kg, "full_knowledge_graph.png")

    # ==================== ERROR HANDLING ==================== #

    except FileNotFoundError:
        print(
            f"Error: Ontology file not found at '{args.ontology_path}'", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred:\n{e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
