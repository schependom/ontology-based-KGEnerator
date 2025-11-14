"""
DESCRIPTION:

    Data structures for
        - Knowledge Graph representation
        - Rules and Constraints
        - Proofs and Backward Chaining

    This replaces the proprietary reldata format with standard Python classes (RRN KGE model)
    and facilitates backward chaining operations.

AUTHOR:

    Vincent Van Schependom
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Union, Optional, Any
from enum import Enum
from rdflib.term import URIRef, Literal
from rdflib.namespace import RDF

# ---------------------------------------------------------------------------- #
#                                     TYPES                                    #
# ---------------------------------------------------------------------------- #

LiteralValue = Union[str, int, float, bool, Literal]
Term = Union[
    "Var", "Individual", "Class", "Relation", "Attribute", URIRef, LiteralValue
]

# ---------------------------------------------------------------------------- #
#                                      KGE                                     #
# ---------------------------------------------------------------------------- #


@dataclass
class Class:
    """Represents a class in the knowledge graph."""

    index: int
    name: str

    def __eq__(self, other):
        if not isinstance(other, Class):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name


@dataclass
class Relation:
    """Represents a relation type in the knowledge graph."""

    index: int
    name: str

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name


@dataclass
class Attribute:
    """Represents an attribute type (datatype property)."""

    index: int
    name: str

    def __eq__(self, other):
        if not isinstance(other, Attribute):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name


@dataclass
class Individual:
    """Represents an individual entity in the knowledge graph."""

    index: int
    name: str

    # We initialize classes as a list, but will store Membership objects here
    classes: List["Membership"] = field(default_factory=list)

    def get_class_memberships(self) -> Set[Class]:
        """Helper to get all classes this individual is a member of."""
        return {m.cls for m in self.classes if m.is_member}

    def __eq__(self, other):
        if not isinstance(other, Individual):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name


@dataclass
class Membership:
    """
    Represents class membership of an individual.
    This fact can be a base fact (proofs=[]) or an inferred fact (proofs=[...]).

    NOTE

        - is_inferred is kept for backward compatibility
        - it should always be true, since we are creating a synthetic datagenerator without seed base facts
    """

    individual: Individual
    cls: Class
    is_member: bool  # True if member, False if explicitly not a member
    # is_inferred: bool  # Still needed for old reldata compatibility

    # Keep track of all proofs leading to this membership fact
    proofs: List["Proof"] = field(default_factory=list)

    @property
    def is_inferred(self) -> bool:
        """A fact is inferred if it has at least one derived proof."""
        if not self.proofs:
            return False
        # Check if any proof is a derived proof (not a base fact leaf)
        return any(p.rule is not None for p in self.proofs)

    @property
    def is_base_fact(self) -> bool:
        """A fact is a base fact if it has no proofs or only a base proof."""
        return not self.is_inferred

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.individual.name, "rdf:type", self.cls.name))

    def to_atom(self) -> "Atom":
        """Converts this fact to a ground Atom (Atom)."""
        return Atom(self.individual, RDF.type, self.cls)

    def __repr__(self) -> str:
        if self.is_member:
            return f"<{self.individual}, memberOf, {self.cls}>"
        else:
            return f"<{self.individual}, ~memberOf, {self.cls}>"


@dataclass
class Triple:
    """
    Represents a relational triple (subject, predicate, object).
    This fact can be a base fact (proofs=[]) or an inferred fact (proofs=[...]).
    """

    subject: Individual
    predicate: Relation
    object: Individual
    positive: bool  # True for positive predicate, False for negated predicate
    # is_inferred: bool  # Still needed for old reldata compatibility

    # Keep track of all proofs leading to this triple fact
    proofs: List["Proof"] = field(default_factory=list)

    @property
    def is_inferred(self) -> bool:
        """A fact is inferred if it has at least one derived proof."""
        if not self.proofs:
            return False
        return any(p.rule is not None for p in self.proofs)

    @property
    def is_base_fact(self) -> bool:
        """A fact is a base fact if it has no proofs or only a base proof."""
        return not self.is_inferred

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.subject.name, self.predicate.name, self.object.name))

    def to_atom(self) -> "Atom":
        """Converts this fact to a ground Atom (Atom)."""
        return Atom(self.subject, self.predicate, self.object)

    def __repr__(self) -> str:
        if self.positive:
            return f"<{self.subject}, {self.predicate}, {self.object}>"
        else:
            return f"<{self.subject}, ~{self.predicate}, {self.object}>"


@dataclass
class AttributeTriple:
    """
    Represents an attribute triple (subject, predicate, value).
    This fact can be a base fact or an inferred fact.
    """

    subject: Individual
    predicate: Attribute  # E.g. age, height
    value: LiteralValue  # The literal value

    # Keep track of all proofs leading to this attribute triple fact
    proofs: List["Proof"] = field(default_factory=list)

    @property
    def is_inferred(self) -> bool:
        """A fact is inferred if it has at least one derived proof."""
        if not self.proofs:
            return False
        return any(p.rule is not None for p in self.proofs)

    @property
    def is_base_fact(self) -> bool:
        """A fact is a base fact if it has no proofs or only a base proof."""
        return not self.is_inferred

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.subject.name, self.predicate.name, self.value))

    def to_atom(self) -> "Atom":
        """Converts this fact to a ground Atom (Atom)."""
        return Atom(self.subject, self.predicate, self.value)

    def __repr__(self) -> str:
        return f"<{self.subject}, {self.predicate}, {self.value}>"


@dataclass
class KnowledgeGraph:
    """
    Complete knowledge graph.
    """

    attributes: List[Attribute]
    classes: List[Class]
    relations: List[Relation]
    individuals: List[Individual]
    triples: List[Triple]
    memberships: List[Membership]
    attribute_triples: List[AttributeTriple]

    def print(self) -> None:
        """Prints a summary of the knowledge graph."""
        print("Knowledge Graph Summary:")
        print(f"  Nb of Attributes: {len(self.attributes)}")
        print(f"  Nb of Classes: {len(self.classes)}")
        print(f"  Nb of Relations: {len(self.relations)}")
        print(f"  Nb of Individuals: {len(self.individuals)}")
        print(f"  Nb of Triples: {len(self.triples)}")
        print(f"  Nb of Memberships: {len(self.memberships)}")
        print(f"  Nb of Attribute Triples: {len(self.attribute_triples)}\n")
        print("Triples:")
        for triple in self.triples:
            print(f"    {triple}")
        print("Memberships:")
        for membership in self.memberships:
            print(f"    {membership}")
        print("Attribute Triples:")
        for attr_triple in self.attribute_triples:
            print(f"    {attr_triple}")


@dataclass
class DataType(Enum):
    """
    Specifies what type of data to use in the KGE model.
    This is for handling the data generated by the ASP solver in the original RRN paper.
    """

    INF = 1  # inferred facts
    SPEC = 2  # specific (base) facts
    ALL = 3  # all facts


# ---------------------------------------------------------------------------- #
#                               BACKWARD CHAINING                              #
# ---------------------------------------------------------------------------- #

"""
EXAMPLE

Rules:
parent(X,Y), parent(Y,Z) -> grandparent(X,Z)    (Rule1)
child(Y,X) -> parent(X,Y)                       (Rule2)

We select the first rule. We see that the head is grandparent(X,Z) and the body is parent(X,Y), parent(Y,Z).
-> goal = Atom(Var('A'), Relation('grandparent'), Var('B'))
-> premises = [Atom(Var('A'), Relation('parent'), Var('C')),
               Atom(Var('C'), Relation('parent'), Var('B'))]
-> rule = ExecutableRule('Rule1', goal, premises)

Now, we want to generate a proof for grandparent(X,Z) and we want to generate
individuals along the way.
"""


@dataclass(frozen=True)  # Variables are immutable and hashable
class Var:
    """Represents a variable in a rule, e.g., 'X' or 'Y'."""

    name: str

    def __repr__(self):
        return f"{self.name}"


@dataclass(frozen=True)  # Atoms are immutable and hashable
class Atom:
    """
    Represents a triple pattern, e.g. (Var('X'), rdf:type, Class('Person')).
    Can represent a ground atom (if no vars) or a pattern (with vars).
    """

    subject: Term
    predicate: Term
    object: Term

    def is_ground(self) -> bool:
        """
        Checks if the goal pattern is ground (no variables).
        """
        return not (
            isinstance(self.subject, Var)
            or isinstance(self.predicate, Var)
            or isinstance(self.object, Var)
        )

    def substitute(self, substitution: Dict[Var, Term]) -> "Atom":
        """
        Applies a variable substitution to this pattern.
        """
        return Atom(
            subject=substitution.get(self.subject, self.subject),
            predicate=substitution.get(self.predicate, self.predicate),
            object=substitution.get(self.object, self.object),
        )


@dataclass
class ExecutableRule:
    """
    Represents an executable rule derived from an ontology axiom.

    E.g., rdfs:subClassOf(ClassA, ClassB) becomes:
    Conclusion: (Var('X'), rdf:type, ClassB)
    Premises:   [(Var('X'), rdf:type, ClassA)]
    """

    name: str
    conclusion: Atom
    premises: List[Atom]

    def __repr__(self):
        prem_str = ", ".join(map(str, self.premises))
        return f"{prem_str} -> {self.conclusion}  ({self.name})"

    def __hash__(self):
        return hash(self.name)

    def is_recursive(self) -> bool:
        """
        A rule is recursive if any atom in its body (premises)
        shares the same "key" (e.g., predicate or class) as the head.
        """

        head = self.conclusion
        head_pred = head.predicate

        # Get the "type" term (class for rdf:type, or None)
        # This handles cases like A(x) -> B(x)
        head_type_term = None
        if head_pred == RDF.type and isinstance(head.object, (Class, Var)):
            head_type_term = head.object

        for premise in self.premises:
            # 1. Check predicate match
            if premise.predicate == head_pred:
                # 2. If predicate is rdf:type, check class match
                if head_type_term is not None:
                    # e.g., A(X) ... -> A(Y)
                    if premise.object == head_type_term:
                        return True
                # 3. If predicate is not rdf:type, just predicate match is enough
                else:
                    # e.g., P(X,Y) ... -> P(X,Z)
                    return True
        return False


@dataclass
class Constraint:
    """
    Represents a constraint that must not be violated.
    E.g., owl:disjointWith(ClassA, ClassB)
    """

    name: str
    constraint_type: URIRef  # e.g., OWL.disjointWith
    terms: List[Term]  # e.g., [Class('ClassA'), Class('ClassB')]

    def __repr__(self):
        return f"Constraint(name={self.name}, type={self.constraint_type}, terms={self.terms})"


@dataclass(frozen=True)  # Proofs are immutable and hashable
class Proof:
    """
    Represents a proof tree for a single GROUND goal (Atom).

    A proof is either for
        - a base fact
            -> if no rules can be applied to prove a goal
            -> leaf in the proof tree
            -> rule=None
        - a derived fact
            -> if a rule was applied to prove the goal
            -> node in the proof tree
            -> rule=ExecutableRule, sub_proofs=[Proof, ...]
    """

    # The ground atom this proof satisfies.
    goal: Atom

    # The rule (ExecutableRule) whose conclusion (ExecutableRule.conclusion)
    # was unified with the goal (grounded Atom).
    # If 'None', this proof represents a base fact (a leaf in the tree).
    rule: Optional[ExecutableRule] = None

    # The list of proofs for the premises of the rule.
    # Must be empty if rule is None.
    sub_proofs: Tuple["Proof", ...] = field(default_factory=tuple)
    # We use a Tuple instead of List to make Proof hashable

    # recursive_use_counts
    #   -> tracks {rule_name: count} for recursive rules.
    #   -> is an immutable, hashable set
    recursive_use_counts: frozenset[Tuple[str, int]] = field(default_factory=frozenset)
    # field() returns an empty frozenset

    def __post_init__(self):
        # A base fact proof cannot have sub-proofs
        if self.rule is None and self.sub_proofs:
            raise ValueError("Base fact proof (rule=None) cannot have sub-proofs.")

        # A derived fact proof must have the same number of sub-proofs as premises
        # in the rule it tries to prove.
        if self.rule is not None and len(self.sub_proofs) != len(self.rule.premises):
            raise ValueError(
                f"Proof for rule '{self.rule.name}' must have "
                f"{len(self.rule.premises)} sub-proofs, but "
                f"{len(self.sub_proofs)} were given."
            )

        # The goal of a proof must be ground.
        if not self.goal.is_ground():
            raise ValueError(f"Proof goal '{self.goal}' must be a ground atom.")

    def is_base_fact(self) -> bool:
        """
        Checks if this proof represents a base fact (leaf).
        """
        return self.rule is None

    def get_base_facts(self) -> Set[Atom]:
        """
        Traverses the proof tree and returns the set of all base facts (leaves) this proof depends on.
        """

        # This is a base fact (a leaf)
        if self.rule is None:
            return {self.goal}

        # Derived fact: gather base facts from sub-proofs
        base_facts: Set[Atom] = set()
        for sp in self.sub_proofs:
            base_facts.update(sp.get_base_facts())
        return base_facts

    def get_recursion_depth(self, rule: ExecutableRule) -> int:
        """
        Gets the number of times the given recursive rule was used in this proof path.
        """
        for name, count in self.recursive_use_counts:
            if name == rule.name:
                return count
        return 0

    def get_max_recursion_depth(self) -> int:
        """
        Gets the maximum depth of any recursive rule in this proof.
        """
        # Check if there are any recursive rules used
        if not self.recursive_use_counts:
            return 0

        # If there are, return the max count
        return max(count for _, count in self.recursive_use_counts)

    @staticmethod
    def create_base_proof(atom: Atom) -> "Proof":
        """
        Creates a proof for a base fact (a leaf).
        """
        # Goal must be a ground atom
        if not atom.is_ground():
            raise ValueError("Base fact proof must be for a ground atom.")

        # Return a proof with no rule and no sub-proofs for the base fact (ground Atom)
        return Proof(goal=atom, rule=None, sub_proofs=tuple())

    @staticmethod
    def create_derived_proof(
        goal: Atom, rule: ExecutableRule, sub_proofs: List["Proof"]
    ) -> "Proof":
        """
        Creates a proof for a derived fact (a node), tracking recursion.
        """
        # Goal must be a ground atom
        if not goal.is_ground():
            raise ValueError("Derived proof goal must be a ground atom.")

        # Combine recursive_use_counts from sub-proofs
        new_counts: Dict[str, int] = {}
        for sp in sub_proofs:
            for name, count in sp.recursive_use_counts:
                # We take the MAX depth from any sub-proof branch for each rule
                # to make sure that the depth doesn't get underestimated.
                new_counts[name] = max(new_counts.get(name, 0), count)

        # Update this rule's count if it's recursive
        if rule.is_recursive():
            name = rule.name
            new_counts[name] = new_counts.get(name, 0) + 1

        # Return the proof
        return Proof(
            goal=goal,
            rule=rule,
            sub_proofs=tuple(sub_proofs),
            recursive_use_counts=frozenset(new_counts.items()),
        )
