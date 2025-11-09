"""
DESCRIPTION

    Pure Python Backward-Chaining Knowledge Graph Generator.

WORKFLOW

    - Shuffle the rules
    - Keep track of generated facts' (hash-set) hashes to avoid duplicates
    - For each rule, try to cover it by backward-chaining
        - Each proof has its own variable bindings
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
            base_fact_prob:     Probability of generating a base fact in the proof tree.
            verbose:            Whether to enable verbose output for debugging.
        """

        # Set seed
        random.seed(seed)

        self.verbose = verbose

        # ------------- KEEP TRACK OF NUMBER OF PROOFS ATTEMPTED PER RULE ------------ #

        self.nb_proofs_attempted: defaultdict[str, int] = defaultdict(int)

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

    # ---------------------------------------------------------------------------- #
    #                             CREATE/GET INDIVIDUAL                            #
    # ---------------------------------------------------------------------------- #

    def _get_or_create_individual(
        self,
    ) -> Individual:
        """
        Creates a new individual or reuses an existing one from the KG with a certain probability.
        """
        # Reuse existing individual from KG
        if self.kg.individuals and random.random() < self.reuse_prob:
            if self.verbose:
                if self.verbose:
                    print(
                        "Reusing existing individual due to reuse probability.",
                        file=sys.stderr,
                    )
            return random.choice(self.kg.individuals)

        # Create a new one
        self.ind_counter += 1
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
                new_ind = self._get_or_create_individual()
                bindings[term] = new_ind
                return new_ind

            # Not bound and not allowed to create
            else:
                print(
                    f"Warning: Var {term} not bound and create_new=False.",
                    file=sys.stderr,
                )
                return None

        # --------------------------- Not individual or var -------------------------- #

        else:
            # Term can be Var, Individual, Class, Relation, Attribute or URIRef
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
        Checks that a fact is does not violate constraints.
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
            return True  # Valid, because it already exists

        # 2. On-the-fly constraint checking
        if not self._satisfies_constraints(fact, proof_being_built=proof_being_built):
            # print(f"CONSTRAINT VIOLATION on {fact}")
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
        Also checks against facts in the proof_fact_cache for chain-local constraints.

        Args:
            new_fact:           The new fact to check.
            proof_fact_cache:    A set of fact hashes for the current chain-being-built.
        Returns:
            True if no constraints are violated, False otherwise.
        """

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

        # Loop over all constraints <A, P, B>
        # self = <A', P', B'>
        for constraint in self.ontology.constraints:
            #
            # -------------------------- owl:disjointWith(A, B) -------------------------- #

            if constraint.constraint_type == OWL.disjointWith:
                class_a, class_b = constraint.terms

                # The fact HAS to be a Membership to be have constraint_type disjointWith
                if isinstance(new_fact, Membership):
                    # Get the individual from the Membership, which has properties [individual, cls, is_member, is_inferred]
                    ind = new_fact.individual

                    # Get all classes from main KG
                    ind_classes = ind.get_class_memberships()

                    # Get classes from chain-being-built
                    for cls in proof_being_built:
                        if isinstance(cls, Membership) and cls.individual == ind:
                            ind_classes.append(cls.cls)

                    # Check for disjoint violation in KG and chain being built
                    if new_fact.cls == class_a and class_b in ind_classes:
                        return False
                    if new_fact.cls == class_b and class_a in ind_classes:
                        return False

            # ----------------------- owl:FunctionalProperty(P) ------------------------ #

            # For a FunctionalProperty P, an individual can have only one value for P.
            # If <A, P, B> exists, we cannot add <A, P, B'> with B != B'.

            elif constraint.constraint_type == OWL.FunctionalProperty:
                prop = constraint.terms[0]
                subj = new_fact.subject

                # Check Relation
                if (
                    isinstance(prop, Relation)  # should always be true
                    and isinstance(new_fact, Triple)
                    and new_fact.predicate == prop  # check if P=P'
                ):
                    # subj=A, prop=P
                    for existing_triple in get_all_triples(subj, prop):
                        # existing_triple = <A, P, B>
                        # new_fact = <A, P, B'>
                        # Check if B != B'
                        if existing_triple.object != new_fact.object:
                            # Violation if B != B'
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

                # Check against chain-being-built
                if (
                    isinstance(prop, Relation)
                    and isinstance(new_fact, Triple)
                    and new_fact.predicate == prop
                ):
                    subj = new_fact.subject
                    for existing_triple in get_all_triples(subj, prop):
                        if (
                            existing_triple.subject == subj
                            and existing_triple.predicate == prop
                            and existing_triple.object != new_fact.object
                        ):
                            return False

            # --------------------- rdfs:range (for DatatypeProperty) -------------------- #

            # DatatypeProperties have rdfs:range constraints specifying the data type of their values.

            # So, assume the constraint we are considering is
            #       <P, rdfs:range, D>,
            # then the value of any AttributeTriple with predicate P must be of data type D.

            elif constraint.constraint_type == RDFS.range and isinstance(
                new_fact, AttributeTriple
            ):
                # If there are less than 2 terms in the constraint, it is malformed
                if len(constraint.terms) < 2:
                    print(
                        f"Warning: Malformed rdfs:range constraint: {constraint}",
                        file=sys.stderr,
                    )

                # Get the property and expected data type
                prop, data_type_uri = constraint.terms
                # New fact has to be an AttributeTriple with predicate P
                if new_fact.predicate == prop:
                    # Get the value of the attribute triple
                    value = new_fact.value
                    # Check the data type
                    if data_type_uri == XSD.string and not isinstance(value, str):
                        return False
                    if data_type_uri == XSD.integer and not isinstance(value, int):
                        return False
                    if data_type_uri == XSD.float and not isinstance(value, float):
                        return False
                    if data_type_uri == XSD.boolean and not isinstance(value, bool):
                        return False

            # -------------------- rdfs:range (for ObjectTypeProperty) ------------------- #

            # ObjectTypeProperties have rdfs:range constraints specifying the class of their object.
            elif constraint.constraint_type == RDFS.range and isinstance(
                new_fact, Triple
            ):
                # If there are less than 2 terms in the constraint, it is malformed
                if len(constraint.terms) < 2:
                    print(
                        f"Warning: Malformed rdfs:range constraint: {constraint}",
                        file=sys.stderr,
                    )

                # Get the property and expected class
                prop, expected_class = constraint.terms

                # New fact <A, P, B> has to be a Triple with P=prop
                if new_fact.predicate == prop:
                    obj = new_fact.object
                    # Check if the object's classes include the expected class
                    obj_classes = obj.get_class_memberships()
                    if expected_class not in obj_classes:
                        return False

            # ------------------------ owl:IrreflexiveProperty(P) ------------------------ #

            elif constraint.constraint_type == OWL.IrreflexiveProperty:
                prop = constraint.terms[0]
                if isinstance(new_fact, Triple) and new_fact.predicate == prop:
                    if new_fact.subject == new_fact.object:
                        print(
                            f"VIOLATION: Irreflexive property {prop.name} on {new_fact.subject.name}"
                        )
                        return False  # Violation

                    # If new fact is <A, hasMother, A> and the constraint
                    #   <hasParent, owl:IrreflexiveProperty> holds,
                    # we want to check <hasMother, owl:IrreflexiveProperty> as well,
                    # which doesn't hold for <A, hasMother, A>!
                    for subproperty in self.ontology.get_subproperties(prop):
                        if (
                            isinstance(new_fact, Triple)
                            and new_fact.predicate == subproperty
                            and new_fact.subject == new_fact.object
                        ):
                            print(
                                f"VIOLATION: Irreflexive property {subproperty.name} on {new_fact.subject.name}! The superproperty {prop.name} is irreflexive."
                            )
                            return False

        # TODO check other constraints.

        return True  # No violations found

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
            num_proofs_per_rule:    Number of attempts to generate a chain per rule.

        Terminology:

            -   A "chain" is a set of facts (base + inferred) that together cover a rule.

                E.g. to cover rule
                    R: (A, grandparentOf, C) :- (A, parentOf, B), (B, parentOf, C)
                we need to start from the goal (A, grandparentOf, C) try to prove the 2 premises (A, parentOf, B) and (B, parentOf, C).

                In this context, a chain would be the set of facts {(A, grandparentOf, C), (A, parentOf, B), (B, parentOf, C)}.
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
            ## TODO I think this is no longer needed
            # if rule.name in self.covered_rules:
            #     continue

            # ---------------------------------------------------------------------------- #
            #                                START NEW PROOF                               #
            # ---------------------------------------------------------------------------- #

            # num_proofs_per_rule attempts to cover this rule
            for proof_no in range(num_proofs_per_rule):
                #
                if self.verbose:
                    print(
                        f"----------------\nAttempting to cover rule: {rule.name} (proof nb. {proof_no + 1}/{num_proofs_per_rule})",
                        file=sys.stderr,
                    )

                # # Stop if already covered
                # if rule.name in self.covered_rules:
                #     break

                # Keep track of variable bindings.
                # A variable can be bound to an Individual or a literal value.
                #
                # Per new chain (renamed 'proof') attempt, we start with empty bindings.
                # So, to make a Prolog analogy, we 'rename' variables per chain/proof attempt.
                # A chain is analogous to one (possibly failed) Prolog proof attempt.
                bindings: Dict[Var, Union[Individual, LiteralValue]] = {}

                # Collect all facts that were created when trying to prove the premises
                all_premise_facts: List[Union[Triple, Membership, AttributeTriple]] = []

                # ------------------------ TRY TO SATISFY THE PREMISES ----------------------- #

                all_premises_satisfied = True

                # Loop over premises and try to satisfy them
                for premise in rule.premises:
                    #
                    # Try to satisfy the premise using Backward-Chaining given the current bindings
                    success, facts_for_premise = self._generate_goal(
                        premise, max_depth - 1, bindings, allow_base_case=True
                    )

                    # If any premise fails, the whole rule fails
                    if not success:
                        all_premises_satisfied = False
                        if self.verbose:
                            print(
                                f"Failed to satisfy premise {premise} for rule {rule.name}. Trying next proof...",
                                file=sys.stderr,
                            )
                        # Break out of the premise loop (try next proof (and thus new bindings))
                        # Because not all premises are satisfied, we continue to the next proof attempt
                        break

                    # Collect all premise facts
                    all_premise_facts.extend(facts_for_premise)

                # -------------------------- ALL PREMISES SATISFIED -------------------------- #

                if all_premises_satisfied:
                    #
                    # All premises are satisfied, now create the conclusion
                    inferred_fact = self._create_inferred_fact(
                        rule.conclusion, bindings, is_inferred=True
                    )

                    if not inferred_fact:
                        if self.verbose:
                            print(
                                f"Failed to create inferred fact for rule {rule.name}. Trying next proof...",
                                file=sys.stderr,
                            )
                        # Go to the next proof attempt (next for loop iteration)
                        continue

                    # Combine all facts: premises + inferred conclusion
                    generated_facts = all_premise_facts + [inferred_fact]

                    # Check all generated facts for duplicates/constraints
                    all_facts_valid = True

                    # If a fact is valid, it is added to this temporary proof-level cache
                    # to simulate adding it to the KG.
                    proof_cache: Set[int] = set()

                    # ----------------------- CHECK FACTS IN REVERSE ORDER ----------------------- #

                    for fact in reversed(generated_facts):
                        #
                        # _valid_fact() checks both against the KG,
                        # as well as the proof being built
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

                    self.nb_proofs_attempted[rule.name] = proof_no

                    if all_facts_valid:
                        # Add all generated facts to the KG (in reverse order to respect dependencies)
                        for fact in reversed(generated_facts):
                            # Only add if it's not already in the *main* KG
                            if hash(fact) not in self.kg_fact_cache:
                                self._add_fact_to_kg(fact)

                        # Keep track of covered rules
                        self.covered_rules.add(rule.name)
                        # Successfully covered this rule, move to next rule
                        break

        # ---------------------------------------------------------------------------- #
        #                               END OF GENERATION                              #
        # ---------------------------------------------------------------------------- #

        print("--- Generation Complete ---")
        all_rule_names = {r.name for r in self.ontology.rules}
        uncovered_rule_names = all_rule_names - self.covered_rules

        print(f"Covered {len(self.covered_rules)} rules.")

        if uncovered_rule_names:
            print(f"Not covered rules ({len(uncovered_rule_names)}):")
            for r_name in sorted(list(uncovered_rule_names))[:10]:  # Print first 10
                print(f"  - {r_name}")
            if len(uncovered_rule_names) > 10:
                print(f"  ... and {len(uncovered_rule_names) - 10} more.")

        print(f"Total Individuals: {len(self.kg.individuals)}")
        print(f"Total Triples: {len(self.kg.triples)}")
        print(f"Total Memberships: {len(self.kg.memberships)}")
        print(f"Total Attribute Triples: {len(self.kg.attribute_triples)}")

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
    ) -> Tuple[bool, List[Union[Triple, Membership, AttributeTriple]]]:
        """
        Recursively try to prove the goal by either:
           - finding a rule that concludes the goal, and trying to prove its premises
           - creating a base fact that matches the goal

        Args:
            goal:               The goal pattern to satisfy (this is the premise from the previous level in the proof tree).
            depth:              Remaining recursion depth.
            bindings:           Current variable bindings.
            allow_base_case:    Whether to allow generating base facts

        Returns:
            A tuple (success, generated_facts).
        """
        #
        # Hit depth limit
        if depth <= 0:
            return (False, [])

        # Find rules from the ontology that can conclude this goal
        applicable_rules = [
            r for r in self.ontology.rules if r.conclusion.matches(goal)
        ]

        # if not applicable_rules:
        #     print(
        #         f"No applicable rules found to prove goal {goal} at depth {depth}. However, 'allow_base_case'={allow_base_case}, trying base case if allowed.",
        #         file=sys.stderr,
        #     )

        # Copy bindings to a temp variable for this proof attempt
        temp_bindings = bindings.copy()

        # ------------------- CREATE A BASE FACT WITH PROPABILITY p ------------------ #

        if allow_base_case and random.random() < self.base_fact_prob:
            if self.verbose:
                if applicable_rules:
                    print(
                        f"Found {len(applicable_rules)} applicable rules, but opting for base case, due to probability."
                    )
                else:
                    print(
                        f"No applicable rules found and due to randomness, must create base fact for goal {goal}."
                    )

            # Create base fact
            success, base_fact = self._try_create_base_fact(goal, temp_bindings)

            if success and base_fact:
                # Success.
                # Commit successful bindings back to the parent scope
                bindings.update(temp_bindings)
                return (True, [base_fact])
            else:
                # Failed to create base fact (e.g., constraint or duplicate)
                # => Try the rules instead (if any)
                if self.verbose:
                    print(
                        f"Failed to create base fact for goal {goal}. Trying applicable rules instead.",
                        file=sys.stderr,
                    )

        # Shuffle applicable_rules to introduce randomness
        random.shuffle(applicable_rules)

        # ------------------------- TRY APPLICABLE RULES ---------------------------- #

        # Loop over all applicable rules to prove the goal
        for rule in applicable_rules:
            #
            # Not a base case, must be a rule
            rule_to_try: ExecutableRule = rule

            # Collect all premise facts generated when trying to prove the premises
            all_premise_facts: List[Union[Triple, Membership, AttributeTriple]] = []
            all_premises_satisfied = True

            # Loop over premises and try to satisfy them
            for premise in rule_to_try.premises:
                #
                # Try to prove the premise using BC on a lower depth
                success, facts_for_premise = self._generate_goal(
                    premise, depth - 1, temp_bindings, allow_base_case=True
                )

                # Check if we succeeded
                if not success:
                    all_premises_satisfied = False

                    # We can't prove the premise, so we can't use this rule to prove the goal.
                    # -> Stop the premise loop.
                    # -> Because all_premises_satisfied is now False, we will try the next applicable rule.
                    break

                # Collect all premise facts
                all_premise_facts.extend(facts_for_premise)

            # Goal proven if all premises are satisfied
            if all_premises_satisfied:
                #
                # Create inferred fact that we were trying to prove (the goal)
                inferred_fact = self._create_inferred_fact(
                    rule_to_try.conclusion, temp_bindings, is_inferred=True
                )

                # Check if the creation of the inferred fact was successful
                if inferred_fact:
                    #
                    # Commit successful bindings to previous level of the proof tree (parent scope)
                    bindings.update(temp_bindings)

                    # Return all generated facts for this goal
                    return (True, all_premise_facts + [inferred_fact])

                else:
                    # Failed to create inferred fact
                    all_premises_satisfied = False

            # else: try the next applicable rule (after `break`: not all premises satisfied)

        # If we get here, we failed to prove the goal
        # because there were no applicable rules
        # or all attempts failed
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

        # Resolve subject
        subject = self._resolve_term(goal.subject, bindings, create_new=True)
        if not subject:
            # Unable to resolve subject (not bound and cannot create)
            return (False, None)

        # ------------------------------ CREATE NEW FACT ----------------------------- #

        #### CLASS MEMBERSHIP ####
        if goal.predicate == RDF.type and isinstance(goal.object, Class):
            new_fact = Membership(
                individual=subject,
                cls=goal.object,
                is_member=True,
                is_inferred=False,  # Base fact
            )

        #### RELATION ####
        elif isinstance(goal.predicate, Relation):
            object_ind = self._resolve_term(goal.object, bindings, create_new=True)
            if not object_ind:
                return (False, None)

            new_fact = Triple(
                subject=subject,
                predicate=goal.predicate,
                object=object_ind,
                positive=True,
                is_inferred=False,  # Base fact
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
                is_inferred=False,  # Base fact
            )
        else:
            return (False, None)

        # --------------------- check constraints and duplicates --------------------- #

        # We use an empty temp_cache because this is the first fact in a chain.
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
        #
        # Already bound?
        if var in bindings:
            val = bindings[var]
            if isinstance(val, (str, int, float, bool)):
                return val
            else:
                return None

        # Find range constraint for this attribute
        data_type_uri = None
        for constraint in self.ontology.constraints:
            if constraint.constraint_type == RDFS.range and constraint.terms[0] == attr:
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
        # Return the generated value
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
        Creates an inferred fact from a satisfied rule's (all premises satisfied) conclusion.

        Args:
            conclusion:     The conclusion goal pattern from the rule.
            bindings:       Current variable bindings.
            is_inferred:    Whether the created fact is inferred (True) or base (False).

        Returns:
            The created fact, or None if creation failed.
        """

        # Resolve subject based on current bindings
        #       We set create_new=False because in inferred facts,
        #       all variables should already be bound from the premises.
        subject = self._resolve_term(conclusion.subject, bindings, create_new=False)

        # Subject was not bound
        if not subject:
            print(
                f"Warning: Unbound subject variable {conclusion.subject} in conclusion {conclusion}.",
                file=sys.stderr,
            )

            # This can happen if the conclusion var isn't in the premises
            # e.g. (A, type, C) -> (B, type, D)
            # We'll try to create a new one, but this is risky,
            # because it may lead to unbound vars in the conclusion.
            subject = self._resolve_term(conclusion.subject, bindings, create_new=True)
            if not subject:
                print(
                    f"Warning: Unbound subject variable {conclusion.subject} in conclusion {conclusion}."
                )
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

        #### RELATION ####
        elif isinstance(conclusion.predicate, Relation):
            #
            # Resolve object based on current bindings
            object_ind = self._resolve_term(
                conclusion.object, bindings, create_new=False
            )

            if not object_ind:
                # Same as above, unbound object var
                print(
                    f"Warning: Unbound object variable {conclusion.object} in conclusion {conclusion}.",
                    file=sys.stderr,
                )
                # This can happen if the object var isn't in the premises
                # e.g. (A, type, C) -> (B, type, D)
                # We'll try to create a new one, but this is risky,
                # because it may lead to unbound vars in the conclusion.
                object_ind = self._resolve_term(
                    conclusion.object, bindings, create_new=True
                )
                if not object_ind:
                    print(
                        f"Warning: Unbound object variable {conclusion.object} in conclusion {conclusion}."
                    )
                    return None

            # Return the generated relation triple
            return Triple(
                subject=subject,
                predicate=conclusion.predicate,
                object=object_ind,
                positive=True,
                is_inferred=is_inferred,
            )

        #### ATTRIBUTE ####
        elif isinstance(conclusion.predicate, Attribute):
            # In <s, attribute, v>, v can be a Var or a constant value
            value = None

            # Var -> get from bindings
            if isinstance(conclusion.object, Var):
                value = bindings.get(conclusion.object)

            # If not a Var, it must be a constant value
            else:
                value = conclusion.object

            # Couldn't resolve value from bindings
            if value is None:
                print(
                    f"Warning: Unbound value variable {conclusion.object} in conclusion {conclusion}."
                )
                return None

            # Return the generated attribute triple
            return AttributeTriple(
                subject=subject,
                predicate=conclusion.predicate,
                value=value,
                is_inferred=is_inferred,
            )

        # Unknown conclusion type
        else:
            print(
                f"Warning: Unable to create inferred fact for conclusion {conclusion}. Unknown predicate type.",
                file=sys.stderr,
            )
            return None

    # ---------------------------------------------------------------------------- #
    #                   PRINT NUMBER OF PROOFS ATTEMPTED PER RULE                  #
    # ---------------------------------------------------------------------------- #

    def print_proof_attempts_per_rule(self) -> None:
        """
        Prints the number of proof attempts made per rule.
        """
        print("\n--- Number of Proof Attempts per Rule ---")
        for rule_name, attempts in self.nb_proofs_attempted.items():
            print(f"  Rule: {rule_name}, Proof Attempts: {attempts}")
