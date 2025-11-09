"""
DESCRIPTION

    Pure Python Backward-Chaining Knowledge Graph Generator.

WORKFLOW

    - Shuffle the rules
    - Keep track of generated facts' (hash-set) hashes to avoid duplicates
    - For each rule, try to cover it by backward-chaining
        - Each proof has its own variable bindings
        - Track proof structures to ensure diversity
        - If a proof (binding) fails, try again up to N times
        - Keep proof-level hash-set to avoid duplicates within a proof
        - When generating a fact, check constraints against:
            - the main KG (kg_fact_cache)
            - the proof being built (proof_fact_cache)
    - Append hash of each generated fact to the KG-level hash set

AUTHOR

    Vincent Van Schependom
"""

from collections import defaultdict
import random
import sys
from typing import List, Union, Dict, Set, Tuple, Optional, Any
from rdflib.namespace import RDF, OWL, RDFS, XSD

from data_structures import (
    Individual,
    Class,
    Relation,
    Attribute,
    KnowledgeGraph,
    Triple,
    Membership,
    AttributeTriple,
    LiteralValue,
    Var,
    Term,
    Constraint,
    GoalPattern,
    ExecutableRule,
)
from parser import OntologyParser


class BackwardChainingGenerator:
    # ---------------------------------------------------------------------------- #
    #                                     INIT                                     #
    # ---------------------------------------------------------------------------- #

    def __init__(
        self,
        parsed_ontology: OntologyParser,
        seed: Optional[int],
        reuse_prob: float,
        base_fact_prob: float,
        verbose: bool,
    ):
        """
        Initializes the generator with a parsed ontology.

        Args:
            parsed_ontology:    An instance of OntologyParser containing the parsed ontology.
            seed:               Optional random seed for reproducibility.
            reuse_prob:         Probability of reusing an existing individual.
            base_fact_prob:     Base probability of generating a base fact in the proof tree.
            verbose:            Whether to enable verbose output for debugging.
        """

        # Set seed
        random.seed(seed)

        self.verbose = verbose

        # ------------- KEEP TRACK OF NUMBER OF PROOFS ATTEMPTED PER RULE ------------ #

        self.nb_proofs_attempted: defaultdict[str, int] = defaultdict(int)
        self.proof_structures: Dict[str, Set[str]] = defaultdict(set)

        # ----------------------- ONTOLOGY AND KNOWLEDGE GRAPH ----------------------- #

        # Type OntologyParser
        self.ontology = parsed_ontology

        # Set probabilities
        self.reuse_prob = reuse_prob
        self.base_fact_prob = base_fact_prob

        # Initialize empty KG with classes, relations, attributes from ontology
        self.kg = KnowledgeGraph(
            classes=list(self.ontology.classes.values()),
            relations=list(self.ontology.relations.values()),
            attributes=list(self.ontology.attributes.values()),
            individuals=[],
            triples=[],
            memberships=[],
            attribute_triples=[],
        )

        # ----------------------------- HELPER VARIABLES ----------------------------- #

        # ind_counter keeps track of the next individual index to assign
        self.ind_counter = 0

        # A HashSet to quickly check for existing facts
        # -> avoid duplicates and speed up constraint checking
        self.kg_fact_cache: Set[int] = set()

        # ----------------------------------- RULES ---------------------------------- #

        self.rules_to_cover: List[ExecutableRule] = list(self.ontology.rules)
        random.shuffle(self.rules_to_cover)  # Randomize rule order

        self.covered_rules: Set[str] = set()

        # -------------------------- INDEXED CONSTRAINTS ----------------------------- #

        self.constraints_by_property: Dict[str, List[Constraint]] = defaultdict(list)
        self.constraints_by_class: Dict[str, List[Constraint]] = defaultdict(list)
        self.all_constraints: List[Constraint] = []
        self._index_constraints()

        # ---------------------------- GENERATION STATS ------------------------------ #

        self.failed_constraint_checks = 0
        self.duplicate_facts = 0
        self.individuals_created = 0
        self.individuals_reused = 0

    # ---------------------------------------------------------------------------- #
    #                              INDEX CONSTRAINTS                               #
    # ---------------------------------------------------------------------------- #

    def _index_constraints(self):
        """Index constraints by relevant properties and classes for efficient lookup"""
        for constraint in self.ontology.constraints:
            self.all_constraints.append(constraint)

            if constraint.constraint_type == OWL.FunctionalProperty:
                prop = constraint.terms[0]
                self.constraints_by_property[prop.name].append(constraint)

            elif constraint.constraint_type == OWL.disjointWith:
                for cls in constraint.terms:
                    self.constraints_by_class[cls.name].append(constraint)

            elif constraint.constraint_type == RDFS.range:
                if len(constraint.terms) >= 1:
                    prop = constraint.terms[0]
                    self.constraints_by_property[prop.name].append(constraint)

            elif constraint.constraint_type == OWL.IrreflexiveProperty:
                prop = constraint.terms[0]
                self.constraints_by_property[prop.name].append(constraint)

    # ---------------------------------------------------------------------------- #
    #                             CREATE/GET INDIVIDUAL                            #
    # ---------------------------------------------------------------------------- #

    def _get_or_create_individual(
        self,
        required_class: Optional[Class] = None,
    ) -> Individual:
        """
        Creates a new individual or reuses an existing one from the KG with a certain probability.
        If required_class is specified, only reuse individuals of that class.

        Args:
            required_class: If specified, only reuse individuals that are members of this class.
        """
        # Reuse existing individual from KG
        if self.kg.individuals and random.random() < self.reuse_prob:
            candidates = self.kg.individuals

            # Filter by required class if specified
            if required_class:
                candidates = [
                    ind
                    for ind in self.kg.individuals
                    if required_class in ind.get_class_memberships()
                ]

            if candidates:
                if self.verbose:
                    print(
                        f"Reusing existing individual (class filter: {required_class.name if required_class else 'none'})",
                        file=sys.stderr,
                    )
                self.individuals_reused += 1
                return random.choice(candidates)

        # Create a new one
        self.ind_counter += 1
        self.individuals_created += 1
        name = f"ind_{self.ind_counter}"
        ind = Individual(index=self.ind_counter, name=name, classes=[])
        self.kg.individuals.append(ind)
        return ind

    # ---------------------------------------------------------------------------- #
    #                            BIND TERM TO INDIVIDUAL                           #
    # ---------------------------------------------------------------------------- #

    def _resolve_term(
        self,
        term: Term,  # The Term (Var, Individual) to resolve
        bindings: Dict[Var, Any],  # Current variable bindings
        create_new: bool = True,  # Whether to create a new Individual for unbound Vars
        required_class: Optional[Class] = None,  # Type hint for new individuals
    ) -> Optional[Individual]:
        """
        Resolves a term to an Individual.
        """

        # -------------------------------- INDIVIDUAL -------------------------------- #

        if isinstance(term, Individual):
            return term

        # ------------------------------------ VAR ----------------------------------- #

        elif isinstance(term, Var):
            # Already bound
            if term in bindings:
                bound_val = bindings[term]
                if isinstance(bound_val, Individual):
                    # Bound to an Individual
                    return bound_val
                else:
                    # Bound to a non-Individual (e.g., literal)
                    return None

            # Not bound and allowed to create
            elif create_new:
                new_ind = self._get_or_create_individual(required_class=required_class)
                bindings[term] = new_ind
                return new_ind

            # Not bound and not allowed to create
            else:
                if self.verbose:
                    print(
                        f"Warning: Var {term} not bound and create_new=False.",
                        file=sys.stderr,
                    )
                return None

        # --------------------------- Not individual or var -------------------------- #

        else:
            # Term can be Var, Individual, Class, Relation, Attribute or URIRef
            if self.verbose:
                print(
                    f"Warning: Unable to resolve term {term}, because it is not an Individual or Var.",
                    file=sys.stderr,
                )
            return None

    # ---------------------------------------------------------------------------- #
    #                                   ADD FACT                                   #
    # ---------------------------------------------------------------------------- #

    def _add_fact_to_kg(self, fact: Union[Triple, Membership, AttributeTriple]):
        """
        Internal helper to add a fact to the KG and kg_fact_cache.

        Args:
            fact: The fact to add.

        Note:
            - Assumes the fact has already been checked for duplicates/constraints!!
            - Updates the kg_fact_cache accordingly.
        """

        # ------------------------------- UPDATE CACHE ------------------------------- #

        # Update the hashset cache (used to check for duplicates)
        self.kg_fact_cache.add(hash(fact))

        # --------------------------------- ADD TO KG -------------------------------- #

        if isinstance(fact, Membership):
            self.kg.memberships.append(fact)
            fact.individual.classes.append(
                fact  # Update individual's class memberships as well
            )
        elif isinstance(fact, Triple):
            self.kg.triples.append(fact)
        elif isinstance(fact, AttributeTriple):
            self.kg.attribute_triples.append(fact)

    # ---------------------------------------------------------------------------- #
    #                                  CHECK FACT                                  #
    # ---------------------------------------------------------------------------- #

    def _valid_fact(
        self,
        fact: Union[Triple, Membership, AttributeTriple],
        proof_fact_cache: Set[int],
        proof_being_built: List[Union[Triple, Membership, AttributeTriple]] = [],
    ) -> bool:
        """
        Checks that a fact does not violate constraints.
        If it is already in the KG, it is considered valid.

        Uses two caches:
            - main kg_fact_cache (duplicates)
            - a temporary proof_fact_cache for the current proof.

        Args:
            fact:               The fact to check.
            proof_fact_cache:   A set of fact hashes for the current proof-being-built.
        """

        # Hash of the fact to be checked
        fact_hash = hash(fact)

        # Check for duplicates 1) in main KG and 2) in the current chain-being-built = current proof
        if fact_hash in self.kg_fact_cache or fact_hash in proof_fact_cache:
            self.duplicate_facts += 1
            return True  # Valid, because it already exists

        # 2. On-the-fly constraint checking
        if not self._satisfies_constraints(fact, proof_being_built=proof_being_built):
            self.failed_constraint_checks += 1
            return False  # VIOLATION!

        return True

    # ---------------------------------------------------------------------------- #
    #                              CONSTRAINT CHECKING                             #
    # ---------------------------------------------------------------------------- #

    def _satisfies_constraints(
        self,
        new_fact: Union[Triple, Membership, AttributeTriple],
        proof_being_built: List[Union[Triple, Membership, AttributeTriple]] = [],
    ) -> bool:
        """
        Checks if adding `new_fact` would violate any constraint.
        Uses indexed constraints for efficiency.

        Args:
            new_fact:           The new fact to check.
            proof_being_built:  Facts currently being built in this proof.
        Returns:
            True if no constraints are violated, False otherwise.
        """

        # Get relevant constraints based on fact type
        relevant_constraints = []

        if isinstance(new_fact, Triple):
            relevant_constraints.extend(
                self.constraints_by_property.get(new_fact.predicate.name, [])
            )
        elif isinstance(new_fact, Membership):
            relevant_constraints.extend(
                self.constraints_by_class.get(new_fact.cls.name, [])
            )
        elif isinstance(new_fact, AttributeTriple):
            relevant_constraints.extend(
                self.constraints_by_property.get(new_fact.predicate.name, [])
            )

        # Helper to get all triples for a subject+predicate combination
        def get_all_triples(subj, prop):
            # Check already-asserted facts
            for t in self.kg.triples:
                if t.subject == subj and t.predicate == prop:
                    yield t

            # Check chain-being-built
            for t in proof_being_built:
                if isinstance(t, Triple) and t.subject == subj and t.predicate == prop:
                    yield t

        # Helper to get all attributes for a subject+predicate combination
        def get_all_attributes(subj, prop):
            # Check already-asserted attribute triples
            for at in self.kg.attribute_triples:
                if at.subject == subj and at.predicate == prop:
                    yield at

            # Check chain-being-built
            for at in proof_being_built:
                if (
                    isinstance(at, AttributeTriple)
                    and at.subject == subj
                    and at.predicate == prop
                ):
                    yield at

        # ---------------------------------------------------------------------------- #
        #                             CHECK ALL CONSTRAINTS                            #
        # ---------------------------------------------------------------------------- #

        # Loop over relevant constraints
        for constraint in relevant_constraints:
            #
            # -------------------------- owl:disjointWith(A, B) -------------------------- #

            if constraint.constraint_type == OWL.disjointWith:
                class_a, class_b = constraint.terms

                # The fact HAS to be a Membership to be have constraint_type disjointWith
                if isinstance(new_fact, Membership):
                    # Get the individual from the Membership
                    ind = new_fact.individual

                    # Get all classes from main KG
                    ind_classes = ind.get_class_memberships()

                    # Get classes from chain-being-built
                    for cls in proof_being_built:
                        if isinstance(cls, Membership) and cls.individual == ind:
                            ind_classes.add(cls.cls)

                    # Check for disjoint violation in KG and chain being built
                    if new_fact.cls == class_a and class_b in ind_classes:
                        return False
                    if new_fact.cls == class_b and class_a in ind_classes:
                        return False

            # ----------------------- owl:FunctionalProperty(P) ------------------------ #

            elif constraint.constraint_type == OWL.FunctionalProperty:
                prop = constraint.terms[0]
                subj = new_fact.subject

                # Check Relation
                if (
                    isinstance(prop, Relation)
                    and isinstance(new_fact, Triple)
                    and new_fact.predicate == prop
                ):
                    for existing_triple in get_all_triples(subj, prop):
                        if existing_triple.object != new_fact.object:
                            return False

                # Check Attribute
                if (
                    isinstance(prop, Attribute)
                    and isinstance(new_fact, AttributeTriple)
                    and new_fact.predicate == prop
                ):
                    for existing_attr_triple in get_all_attributes(subj, prop):
                        if existing_attr_triple.value != new_fact.value:
                            return False

            # --------------------- rdfs:range (for DatatypeProperty) -------------------- #

            elif constraint.constraint_type == RDFS.range and isinstance(
                new_fact, AttributeTriple
            ):
                if len(constraint.terms) < 2:
                    if self.verbose:
                        print(
                            f"Warning: Malformed rdfs:range constraint: {constraint}",
                            file=sys.stderr,
                        )
                    continue

                prop, data_type_uri = constraint.terms
                if new_fact.predicate == prop:
                    value = new_fact.value
                    if data_type_uri == XSD.string and not isinstance(value, str):
                        return False
                    if data_type_uri == XSD.integer and not isinstance(value, int):
                        return False
                    if data_type_uri == XSD.float and not isinstance(value, float):
                        return False
                    if data_type_uri == XSD.boolean and not isinstance(value, bool):
                        return False

            # -------------------- rdfs:range (for ObjectTypeProperty) ------------------- #

            elif constraint.constraint_type == RDFS.range and isinstance(
                new_fact, Triple
            ):
                if len(constraint.terms) < 2:
                    if self.verbose:
                        print(
                            f"Warning: Malformed rdfs:range constraint: {constraint}",
                            file=sys.stderr,
                        )
                    continue

                prop, expected_class = constraint.terms

                if new_fact.predicate == prop:
                    obj = new_fact.object
                    obj_classes = obj.get_class_memberships()

                    # Also check proof_being_built for class memberships
                    for fact in proof_being_built:
                        if isinstance(fact, Membership) and fact.individual == obj:
                            obj_classes.add(fact.cls)

                    if expected_class not in obj_classes:
                        return False

            # ------------------------ owl:IrreflexiveProperty(P) ------------------------ #

            elif constraint.constraint_type == OWL.IrreflexiveProperty:
                prop = constraint.terms[0]
                if isinstance(new_fact, Triple) and new_fact.predicate == prop:
                    if new_fact.subject == new_fact.object:
                        if self.verbose:
                            print(
                                f"VIOLATION: Irreflexive property {prop.name} on {new_fact.subject.name}"
                            )
                        return False

                    # Check subproperties
                    for subproperty in self.ontology.get_subproperties(prop):
                        if (
                            isinstance(new_fact, Triple)
                            and new_fact.predicate == subproperty
                            and new_fact.subject == new_fact.object
                        ):
                            if self.verbose:
                                print(
                                    f"VIOLATION: Irreflexive property {subproperty.name} on {new_fact.subject.name}! "
                                    f"The superproperty {prop.name} is irreflexive."
                                )
                            return False

        return True  # No violations found

    # ---------------------------------------------------------------------------- #
    #                            CREATE PROOF SIGNATURE                            #
    # ---------------------------------------------------------------------------- #

    def _create_proof_signature(
        self, facts: List[Union[Triple, Membership, AttributeTriple]]
    ) -> str:
        """
        Create a signature representing the structure of the proof.
        This helps ensure we generate diverse proofs, not just duplicate structures.
        """
        sig_parts = []
        for fact in facts:
            if isinstance(fact, Triple):
                sig_parts.append(f"T:{fact.predicate.name}")
            elif isinstance(fact, Membership):
                sig_parts.append(f"M:{fact.cls.name}")
            elif isinstance(fact, AttributeTriple):
                sig_parts.append(f"A:{fact.predicate.name}")
        return "|".join(sorted(sig_parts))

    # ---------------------------------------------------------------------------- #
    #                     PURE BACKWARD-CHAINING GENERATOR                         #
    # ---------------------------------------------------------------------------- #

    def generate(
        self,
        max_depth: int,
        num_proofs_per_rule: int,
    ) -> KnowledgeGraph:
        """
        Main generation loop. Tries to cover all rules via backward-chaining.

        Args:
            max_depth:              Maximum recursion depth for backward-chaining.
            num_proofs_per_rule:    Number of diverse proofs to generate per rule.
        """
        print("--- Starting Generation ---")
        print(f"Targeting {len(self.rules_to_cover)} rules.")

        # Copy the rules to cover
        rules_to_try = self.rules_to_cover.copy()

        # ---------------------------------------------------------------------------- #
        #                                LOOP OVER RULES                               #
        # ---------------------------------------------------------------------------- #

        # Loop over rules
        for rule in rules_to_try:
            successful_proofs = 0
            attempts = 0
            max_attempts = num_proofs_per_rule * 5  # Allow more attempts for diversity

            # ---------------------------------------------------------------------------- #
            #                          TRY MULTIPLE DIVERSE PROOFS                         #
            # ---------------------------------------------------------------------------- #

            while successful_proofs < num_proofs_per_rule and attempts < max_attempts:
                attempts += 1

                if self.verbose:
                    print(
                        f"----------------\nAttempting to cover rule: {rule.name} "
                        f"(proof {successful_proofs + 1}/{num_proofs_per_rule}, attempt {attempts}/{max_attempts})",
                        file=sys.stderr,
                    )

                # ------------- VARY GENERATION STRATEGY BASED ON PROOF NUMBER -------------- #

                # Vary base_fact_prob to explore different proof strategies
                if successful_proofs < num_proofs_per_rule // 3:
                    # First third: prefer shallow proofs (more base facts)
                    proof_base_prob = min(0.4, self.base_fact_prob * 8)
                elif successful_proofs < 2 * num_proofs_per_rule // 3:
                    # Second third: prefer deep proofs (fewer base facts)
                    proof_base_prob = self.base_fact_prob * 0.3
                else:
                    # Last third: normal probability
                    proof_base_prob = self.base_fact_prob

                # Keep track of variable bindings
                bindings: Dict[Var, Union[Individual, LiteralValue]] = {}

                # Collect all facts that were created when trying to prove the premises
                all_premise_facts: List[Union[Triple, Membership, AttributeTriple]] = []

                # ------------------------ TRY TO SATISFY THE PREMISES ----------------------- #

                all_premises_satisfied = True

                # Loop over premises and try to satisfy them
                for premise in rule.premises:
                    # Try to satisfy the premise using Backward-Chaining
                    success, facts_for_premise = self._generate_goal(
                        premise,
                        max_depth - 1,
                        bindings,
                        allow_base_case=True,
                        base_fact_prob_override=proof_base_prob,
                        max_depth_context=max_depth,
                    )

                    # If any premise fails, the whole rule fails
                    if not success:
                        all_premises_satisfied = False
                        if self.verbose:
                            print(
                                f"Failed to satisfy premise {premise} for rule {rule.name}.",
                                file=sys.stderr,
                            )
                        break

                    # Collect all premise facts
                    all_premise_facts.extend(facts_for_premise)

                # -------------------------- ALL PREMISES SATISFIED -------------------------- #

                if all_premises_satisfied:
                    # All premises are satisfied, now create the conclusion
                    inferred_fact = self._create_inferred_fact(
                        rule.conclusion, bindings, is_inferred=True
                    )

                    if not inferred_fact:
                        if self.verbose:
                            print(
                                f"Failed to create inferred fact for rule {rule.name}.",
                                file=sys.stderr,
                            )
                        continue

                    # Combine all facts: premises + inferred conclusion
                    generated_facts = all_premise_facts + [inferred_fact]

                    # Check for duplicate proof structure
                    proof_sig = self._create_proof_signature(generated_facts)
                    if proof_sig in self.proof_structures[rule.name]:
                        if self.verbose:
                            print(
                                "Duplicate proof structure, trying different approach..."
                            )
                        continue

                    # Check all generated facts for duplicates/constraints
                    all_facts_valid = True

                    # If a fact is valid, it is added to this temporary proof-level cache
                    proof_cache: Set[int] = set()

                    # ----------------------- CHECK FACTS IN REVERSE ORDER ----------------------- #

                    for fact in reversed(generated_facts):
                        if not self._valid_fact(
                            fact, proof_cache, proof_being_built=generated_facts
                        ):
                            all_facts_valid = False
                            if self.verbose:
                                print(f"Proof failed constraint check for fact: {fact}")
                            break

                        # Valid, so add to proof cache
                        proof_cache.add(hash(fact))

                    # ------------------------- ALL FACTS VALID, ADD TO KG ------------------------ #

                    if all_facts_valid:
                        # Record successful proof structure
                        self.proof_structures[rule.name].add(proof_sig)
                        successful_proofs += 1

                        # Add all generated facts to the KG
                        for fact in reversed(generated_facts):
                            if hash(fact) not in self.kg_fact_cache:
                                self._add_fact_to_kg(fact)

                        # Mark rule as covered
                        self.covered_rules.add(rule.name)

                        if self.verbose:
                            print(
                                f"Successfully generated proof {successful_proofs} for rule {rule.name}"
                            )

            # Record number of attempts for this rule
            if rule.name in self.covered_rules:
                self.nb_proofs_attempted[rule.name] = attempts

        # ---------------------------------------------------------------------------- #
        #                               END OF GENERATION                              #
        # ---------------------------------------------------------------------------- #

        self._print_generation_summary()
        return self.kg

    # ---------------------------------------------------------------------------- #
    #                    BACKWARD CHAIN DOWN IN THE PROOF TREE                     #
    # ---------------------------------------------------------------------------- #

    def _generate_goal(
        self,
        goal: GoalPattern,
        depth: int,
        bindings: Dict[Var, Any],
        allow_base_case: bool = True,
        base_fact_prob_override: Optional[float] = None,
        max_depth_context: Optional[int] = None,
    ) -> Tuple[bool, List[Union[Triple, Membership, AttributeTriple]]]:
        """
        Recursively try to prove the goal by either:
           - finding a rule that concludes the goal, and trying to prove its premises
           - creating a base fact that matches the goal

        Args:
            goal:                   The goal pattern to satisfy.
            depth:                  Remaining recursion depth.
            bindings:               Current variable bindings.
            allow_base_case:        Whether to allow generating base facts.
            base_fact_prob_override: Override for base_fact_prob (for proof diversity).
            max_depth_context:      Original max depth (for depth-dependent probability).

        Returns:
            A tuple (success, generated_facts).
        """
        # Hit depth limit
        if depth <= 0:
            return (False, [])

        # Calculate depth-dependent base fact probability
        if max_depth_context is None:
            max_depth_context = depth

        depth_ratio = 1 - (depth / max_depth_context)  # 0 at top, 1 at bottom
        base_prob = (
            base_fact_prob_override if base_fact_prob_override else self.base_fact_prob
        )

        # Increase probability of base facts as we go deeper
        adjusted_base_prob = base_prob + (depth_ratio * 0.25)
        adjusted_base_prob = min(0.7, adjusted_base_prob)  # Cap at 70%

        # Find rules from the ontology that can conclude this goal
        applicable_rules = [
            r for r in self.ontology.rules if r.conclusion.matches(goal)
        ]

        # Copy bindings to a temp variable for this proof attempt
        temp_bindings = bindings.copy()

        # ------------------- CREATE A BASE FACT WITH PROBABILITY p ------------------ #

        if allow_base_case and random.random() < adjusted_base_prob:
            if self.verbose:
                if applicable_rules:
                    print(
                        f"Found {len(applicable_rules)} applicable rules, but opting for base case "
                        f"(prob={adjusted_base_prob:.2f}, depth={depth}/{max_depth_context})"
                    )
                else:
                    print(
                        f"No applicable rules found, creating base fact for goal {goal} "
                        f"(depth={depth}/{max_depth_context})"
                    )

            # Create base fact
            success, base_fact = self._try_create_base_fact(goal, temp_bindings)

            if success and base_fact:
                # Commit successful bindings back to the parent scope
                bindings.update(temp_bindings)
                return (True, [base_fact])
            else:
                # Failed to create base fact => Try the rules instead (if any)
                if self.verbose:
                    print(
                        f"Failed to create base fact for goal {goal}. Trying applicable rules instead.",
                        file=sys.stderr,
                    )

        # Shuffle applicable_rules to introduce randomness
        random.shuffle(applicable_rules)

        # ------------------------- TRY APPLICABLE RULES ---------------------------- #

        for rule in applicable_rules:
            rule_to_try: ExecutableRule = rule

            # Collect all premise facts generated when trying to prove the premises
            all_premise_facts: List[Union[Triple, Membership, AttributeTriple]] = []
            all_premises_satisfied = True

            # Loop over premises and try to satisfy them
            for premise in rule_to_try.premises:
                # Try to prove the premise using BC on a lower depth
                success, facts_for_premise = self._generate_goal(
                    premise,
                    depth - 1,
                    temp_bindings,
                    allow_base_case=True,
                    base_fact_prob_override=base_fact_prob_override,
                    max_depth_context=max_depth_context,
                )

                if not success:
                    all_premises_satisfied = False
                    break

                # Collect all premise facts
                all_premise_facts.extend(facts_for_premise)

            # Goal proven if all premises are satisfied
            if all_premises_satisfied:
                # Create inferred fact that we were trying to prove (the goal)
                inferred_fact = self._create_inferred_fact(
                    rule_to_try.conclusion, temp_bindings, is_inferred=True
                )

                if inferred_fact:
                    # Commit successful bindings to previous level of the proof tree
                    bindings.update(temp_bindings)
                    return (True, all_premise_facts + [inferred_fact])

        # If we get here, we failed to prove the goal
        if self.verbose:
            print(f"Failed to prove goal {goal} at depth {depth}.", file=sys.stderr)
        return (False, [])

    # ---------------------------------------------------------------------------- #
    #                                   BASE FACT                                  #
    # ---------------------------------------------------------------------------- #

    def _try_create_base_fact(
        self, goal: GoalPattern, bindings: Dict[Var, Any]
    ) -> Tuple[bool, Optional[Union[Triple, Membership, AttributeTriple]]]:
        """
        Tries to create a single base fact matching the goal.
        """

        # Determine required class for type-aware individual creation
        required_class = None
        if goal.predicate == RDF.type and isinstance(goal.object, Class):
            required_class = goal.object

        # Resolve subject
        subject = self._resolve_term(
            goal.subject, bindings, create_new=True, required_class=required_class
        )
        if not subject:
            return (False, None)

        # ------------------------------ CREATE NEW FACT ----------------------------- #

        #### CLASS MEMBERSHIP ####
        if goal.predicate == RDF.type and isinstance(goal.object, Class):
            new_fact = Membership(
                individual=subject,
                cls=goal.object,
                is_member=True,
                is_inferred=False,
            )

        #### RELATION ####
        elif isinstance(goal.predicate, Relation):
            # Try to get required class from range constraints
            range_class = None
            for constraint in self.constraints_by_property.get(goal.predicate.name, []):
                if (
                    constraint.constraint_type == RDFS.range
                    and len(constraint.terms) >= 2
                ):
                    _, potential_class = constraint.terms
                    if isinstance(potential_class, Class):
                        range_class = potential_class
                        break

            object_ind = self._resolve_term(
                goal.object, bindings, create_new=True, required_class=range_class
            )
            if not object_ind:
                return (False, None)

            new_fact = Triple(
                subject=subject,
                predicate=goal.predicate,
                object=object_ind,
                positive=True,
                is_inferred=False,
            )

        #### ATTRIBUTE ####
        elif isinstance(goal.predicate, Attribute):
            value = self._generate_literal_value(goal.object, goal.predicate, bindings)
            if value is None:
                return (False, None)

            new_fact = AttributeTriple(
                subject=subject,
                predicate=goal.predicate,
                value=value,
                is_inferred=False,
            )
        else:
            return (False, None)

        # --------------------- check constraints and duplicates --------------------- #

        if self._valid_fact(
            new_fact, proof_fact_cache=set(), proof_being_built=[new_fact]
        ):
            return (True, new_fact)
        else:
            return (False, None)

    # ---------------------------------------------------------------------------- #
    #                                CREATE LITERAL                                #
    # ---------------------------------------------------------------------------- #

    def _generate_literal_value(
        self, var: Var, attr: Attribute, bindings: Dict[Var, Any]
    ) -> Optional[LiteralValue]:
        """
        Generates a random literal value for an attribute.
        """
        # Already bound?
        if var in bindings:
            val = bindings[var]
            if isinstance(val, (str, int, float, bool)):
                return val
            else:
                return None

        # Find range constraint for this attribute
        data_type_uri = None
        for constraint in self.constraints_by_property.get(attr.name, []):
            if constraint.constraint_type == RDFS.range and len(constraint.terms) >= 2:
                data_type_uri = constraint.terms[1]
                break

        # Assign a random value of the appropriate type
        value: LiteralValue
        if data_type_uri == XSD.string:
            value = f"{attr.name}_{random.randint(1000, 9999)}"
        elif data_type_uri == XSD.integer:
            value = random.randint(1, 100)
        elif data_type_uri == XSD.float:
            value = round(random.random() * 100.0, 2)
        elif data_type_uri == XSD.boolean:
            value = random.choice([True, False])
        else:
            # Default to string if no range is specified
            value = f"val_{random.randint(100, 999)}"

        # Bind the variable
        bindings[var] = value
        return value

    # ---------------------------------------------------------------------------- #
    #                             CREATE INFERRED FACT                             #
    # ---------------------------------------------------------------------------- #

    def _create_inferred_fact(
        self,
        conclusion: GoalPattern,
        bindings: Dict[Var, Any],
        is_inferred: bool = True,
    ) -> Optional[Union[Triple, Membership, AttributeTriple]]:
        """
        Creates an inferred fact from a satisfied rule's conclusion.

        Args:
            conclusion:     The conclusion goal pattern from the rule.
            bindings:       Current variable bindings.
            is_inferred:    Whether the created fact is inferred (True) or base (False).

        Returns:
            The created fact, or None if creation failed.
        """

        # Resolve subject based on current bindings
        subject = self._resolve_term(conclusion.subject, bindings, create_new=False)

        if not subject:
            if self.verbose:
                print(
                    f"Warning: Unbound subject variable {conclusion.subject} in conclusion {conclusion}.",
                    file=sys.stderr,
                )
            subject = self._resolve_term(conclusion.subject, bindings, create_new=True)
            if not subject:
                return None

        # ---------------------------- CREATE MEMBERSHIP ----------------------------- #

        if conclusion.predicate == RDF.type and isinstance(conclusion.object, Class):
            return Membership(
                individual=subject,
                cls=conclusion.object,
                is_member=True,
                is_inferred=is_inferred,
            )

        # ----------------------------- CREATE TRIPLE ------------------------------- #

        elif isinstance(conclusion.predicate, Relation):
            object_ind = self._resolve_term(
                conclusion.object, bindings, create_new=False
            )

            if not object_ind:
                if self.verbose:
                    print(
                        f"Warning: Unbound object variable {conclusion.object} in conclusion {conclusion}.",
                        file=sys.stderr,
                    )
                object_ind = self._resolve_term(
                    conclusion.object, bindings, create_new=True
                )
                if not object_ind:
                    return None

            return Triple(
                subject=subject,
                predicate=conclusion.predicate,
                object=object_ind,
                positive=True,
                is_inferred=is_inferred,
            )

        #### ATTRIBUTE ####
        elif isinstance(conclusion.predicate, Attribute):
            value = None

            if isinstance(conclusion.object, Var):
                value = bindings.get(conclusion.object)
            else:
                value = conclusion.object

            if value is None:
                if self.verbose:
                    print(
                        f"Warning: Unbound value variable {conclusion.object} in conclusion {conclusion}."
                    )
                return None

            return AttributeTriple(
                subject=subject,
                predicate=conclusion.predicate,
                value=value,
                is_inferred=is_inferred,
            )

        else:
            if self.verbose:
                print(
                    f"Warning: Unable to create inferred fact for conclusion {conclusion}. Unknown predicate type.",
                    file=sys.stderr,
                )
            return None

    # ---------------------------------------------------------------------------- #
    #                         GENERATION SUMMARY & STATISTICS                      #
    # ---------------------------------------------------------------------------- #

    def _print_generation_summary(self) -> None:
        """Print summary of generation process"""
        print("\n--- Generation Complete ---")
        all_rule_names = {r.name for r in self.ontology.rules}
        uncovered_rule_names = all_rule_names - self.covered_rules

        print(f"Covered {len(self.covered_rules)}/{len(all_rule_names)} rules.")

        if uncovered_rule_names:
            print(f"\nNot covered rules ({len(uncovered_rule_names)}):")
            for r_name in sorted(list(uncovered_rule_names))[:10]:
                print(f"  - {r_name}")
            if len(uncovered_rule_names) > 10:
                print(f"  ... and {len(uncovered_rule_names) - 10} more.")

        print(f"\nTotal Individuals: {len(self.kg.individuals)}")
        print(f"  - Created: {self.individuals_created}")
        print(f"  - Reused: {self.individuals_reused}")
        print(f"Total Triples: {len(self.kg.triples)}")
        print(f"Total Memberships: {len(self.kg.memberships)}")
        print(f"Total Attribute Triples: {len(self.kg.attribute_triples)}")

    def print_generation_statistics(self) -> None:
        """Print detailed statistics about the generated KG"""
        print("\n" + "=" * 70)
        print("GENERATION STATISTICS")
        print("=" * 70)

        # Rule coverage details
        print("\n RULE COVERAGE")
        print(f"  Covered: {len(self.covered_rules)}/{len(self.ontology.rules)}")

        if self.covered_rules:
            print("\n  Proofs generated per covered rule:")
            for rule_name in sorted(self.covered_rules):
                num_proofs = len(self.proof_structures.get(rule_name, set()))
                attempts = self.nb_proofs_attempted.get(rule_name, 0)
                print(
                    f"    {rule_name}: {num_proofs} diverse proofs ({attempts} attempts)"
                )

        # Connectivity analysis
        print("\n GRAPH CONNECTIVITY")
        connected_individuals = set()
        for triple in self.kg.triples:
            connected_individuals.add(triple.subject)
            connected_individuals.add(triple.object)

        isolated = len(self.kg.individuals) - len(connected_individuals)
        connectivity_ratio = (
            len(connected_individuals) / len(self.kg.individuals) * 100
            if self.kg.individuals
            else 0
        )
        print(
            f"  Connected individuals: {len(connected_individuals)}/{len(self.kg.individuals)} ({connectivity_ratio:.1f}%)"
        )
        print(f"  Isolated individuals: {isolated}")

        # Fact distribution
        all_facts = self.kg.triples + self.kg.memberships + self.kg.attribute_triples
        base_facts = [f for f in all_facts if not f.is_inferred]
        inferred_facts = [f for f in all_facts if f.is_inferred]

        print("\n FACT DISTRIBUTION")
        print(f"  Total facts: {len(all_facts)}")
        print(
            f"  Base facts: {len(base_facts)} ({len(base_facts) / len(all_facts) * 100:.1f}%)"
        )
        print(
            f"  Inferred facts: {len(inferred_facts)} ({len(inferred_facts) / len(all_facts) * 100:.1f}%)"
        )

        # Relation usage
        relation_counts = defaultdict(int)
        for triple in self.kg.triples:
            relation_counts[triple.predicate.name] += 1

        if relation_counts:
            print("\n RELATION USAGE")
            for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
                print(f"  {rel}: {count}")

        # Class membership distribution
        class_counts = defaultdict(int)
        for membership in self.kg.memberships:
            class_counts[membership.cls.name] += 1

        if class_counts:
            print("\n CLASS MEMBERSHIP DISTRIBUTION")
            for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"  {cls}: {count}")

        # Attribute usage
        attribute_counts = defaultdict(int)
        for attr_triple in self.kg.attribute_triples:
            attribute_counts[attr_triple.predicate.name] += 1

        if attribute_counts:
            print("\n ATTRIBUTE USAGE")
            for attr, count in sorted(attribute_counts.items(), key=lambda x: -x[1]):
                print(f"  {attr}: {count}")

        # Generation efficiency
        print("\n GENERATION EFFICIENCY")
        print(f"  Failed constraint checks: {self.failed_constraint_checks}")
        print(f"  Duplicate facts encountered: {self.duplicate_facts}")

        total_attempts = sum(self.nb_proofs_attempted.values())
        if total_attempts > 0:
            success_rate = len(self.covered_rules) / len(self.ontology.rules) * 100
            print(f"  Total proof attempts: {total_attempts}")
            print(f"  Success rate: {success_rate:.1f}%")

        print("\n" + "=" * 70)

    def print_proof_attempts_per_rule(self) -> None:
        """
        Prints the number of proof attempts made per rule (legacy method).
        """
        print("\n--- Number of Proof Attempts per Rule ---")
        for rule_name, attempts in self.nb_proofs_attempted.items():
            num_proofs = len(self.proof_structures.get(rule_name, set()))
            print(
                f"  Rule: {rule_name}, Diverse Proofs: {num_proofs}, Attempts: {attempts}"
            )
