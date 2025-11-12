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
from typing import List, Set, Union, Optional, Any
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
        """A fact is inferred if it has at least one proof."""
        return bool(self.proofs)

    @property
    def is_base_fact(self) -> bool:
        """A fact is a base fact if it has no proofs."""
        return not self.proofs

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.individual.name, "rdf:type", self.cls.name))


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
        """A fact is inferred if it has at least one proof."""
        return bool(self.proofs)

    @property
    def is_base_fact(self) -> bool:
        """A fact is a base fact if it has no proofs."""
        return not self.proofs

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.subject.name, self.predicate.name, self.object.name))


@dataclass
class AttributeTriple:
    """
    Represents an attribute triple (subject, predicate, value).
    This fact can be a base fact (proofs=[]) or an inferred fact (proofs=[...]).
    """

    subject: Individual
    predicate: Attribute  # E.g. age, height
    value: LiteralValue  # The literal value
    # is_inferred: bool  # Still needed for old reldata compatibility

    # Keep track of all proofs leading to this attribute triple fact
    proofs: List["Proof"] = field(default_factory=list)

    @property
    def is_inferred(self) -> bool:
        """A fact is inferred if it has at least one proof."""
        return bool(self.proofs)

    @property
    def is_base_fact(self) -> bool:
        """A fact is a base fact if it has no proofs."""
        return not self.proofs

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.subject.name, self.predicate.name, self.value))


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


# A "term" can be a variable, or a concrete Class/Relation/Individual
@dataclass
class Var:
    """Represents a variable in a rule, e.g., 'X' or 'Y'."""

    name: str

    # TODO check if we need more complex variable handling,
    # e.g. for each proof: renaming variables to avoid clashes

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Var):
            return False
        return self.name == other.name


@dataclass
class GoalPattern:
    """
    Represents a triple pattern, e.g. (Var('X'), rdf:type, Class('Person')).
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

    # The match logic should be implemented in the backward chaining engine
    # because it needs to handle variable unifications and substitutions.
    #
    # So, matches() and unify() methods are omitted here.


@dataclass
class ExecutableRule:
    """
    Represents an executable rule derived from an ontology axiom.

    E.g., rdfs:subClassOf(ClassA, ClassB) becomes:
    Conclusion: (Var('X'), rdf:type, ClassB)
    Premises:   [(Var('X'), rdf:type, ClassA)]
    """

    name: str
    conclusion: GoalPattern
    premises: List[GoalPattern]

    def __repr__(self):
        return f"Rule(name={self.name}, conc={self.conclusion}, prem={self.premises})"

    def is_recursive(self) -> bool:
        """
        Checks if the rule is (simply) recursive.
        This is a basic check: does a predicate in the head
        also appear in the body?

        Note: This doesn't catch mutual recursion.
        """
        head_predicate = self.conclusion.predicate
        if isinstance(head_predicate, Var):
            return True  # Hard to tell, assume yes

        for premise in self.premises:
            if premise.predicate == head_predicate:
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

    # 2 terms, namely A and B, for disjointWith <A, disjointWith, B>,
    # 1 term, namely A, for FunctionalProperty P in <A, P, B>,
    # 2 terms, namely P and D, for rdfs:range (<P, rdfs:range, D>) on AttributeTriples
    # 1 term, namely P, for IrreflexiveProperty (P) for <P, rdf:type, owl:IrreflexiveProperty>

    def __repr__(self):
        return f"Constraint(name={self.name}, type={self.constraint_type}, terms={self.terms})"


"""
EXAMPLE

Rules:
parent(X,Y), parent(Y,Z) -> grandparent(X,Z)    (Rule1)
child(Y,X) -> parent(X,Y)                       (Rule2)

We select the first rule. We see that the head is grandparent(X,Z) and the body is parent(X,Y), parent(Y,Z).
-> goal = GoalPattern(Var('A'), Relation('grandparent'), Var('B'))
-> premises = [GoalPattern(Var('A'), Relation('parent'), Var('C')),
               GoalPattern(Var('C'), Relation('parent'), Var('B'))]
-> rule = ExecutableRule('Rule1', goal, premises)

Now, we want to generate a proof for grandparent(X,Z) and we want to generate
individuals along the way.
"""


@dataclass
class Proof:
    """
    Represents a proof tree for a single goal GoalPattern.
    """

    # The atom this proof satisfies
    goal: GoalPattern

    # The rule whose conclusion was unified with the goal
    rule: ExecutableRule

    # The list of proofs for the premises of the rule.
    # This is empty if it's a base fact.
    sub_proofs: List["Proof"]

    # TODO
