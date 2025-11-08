"""
DESCRIPTION:

    Data structures for
        - Knowledge Graph representation
        - Rules and Constraints
        - Terms and Variables

    This replaces the proprietary reldata format with standard Python classes (RRN KGE model)
    and facilitates backward chaining operations.

AUTHOR:

    Vincent Van Schependom
"""

from dataclasses import dataclass, field
from typing import List, Set, Union
from enum import Enum
from rdflib.term import URIRef

# ---------------------------------------------------------------------------- #
#                                      KGE                                     #
# ---------------------------------------------------------------------------- #


LiteralValue = Union[str, int, float, bool]


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
    """Represents class membership of an individual."""

    individual: Individual
    cls: Class
    is_member: bool  # True if member, False if explicitly not a member
    is_inferred: bool  # True if this is an inferred fact (vs. a base fact)

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.individual.name, "rdf:type", self.cls.name))


@dataclass
class Triple:
    """Represents a relational triple (subject, predicate, object)."""

    subject: Individual
    predicate: Relation
    object: Individual
    positive: bool  # True for positive predicate, False for negated predicate
    is_inferred: bool  # True if this is an inferred fact (vs. a base fact)

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.subject.name, self.predicate.name, self.object.name))


@dataclass
class AttributeTriple:
    """Represents an attribute triple (subject, predicate, value)."""

    subject: Individual
    predicate: Attribute  # E.g. age, height
    value: LiteralValue  # The literal value
    is_inferred: bool  # True if this is an inferred fact (vs. a base fact)

    def __hash__(self):
        # A fact is defined by its content
        return hash((self.subject.name, self.predicate.name, self.value))


@dataclass
class KnowledgeGraph:
    """
    Complete knowledge graph containing
    classes,            (class index, class name)
    relations,          (relation index, relation name)
    individuals,        (individual index, individual name)
    triples,            (subject, predicate, object, positive, is_fact)
    memberships,        (individual, class, is_member, is_fact)
    attributes,         (attribute index, attribute name)                   -> e.g. age, height
    attribute_triples,  (subject, predicate, value, is_fact)                -> e.g. (John, age, 30)
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

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Var):
            return False
        return self.name == other.name


# type
Term = Union[Var, Individual, Class, Relation, Attribute, URIRef]


@dataclass
class GoalPattern:
    """
    Represents a triple pattern, e.g. (Var('X'), rdf:type, Class('Person')).
    """

    subject: Term
    predicate: Term
    object: Term

    def matches(self, goal: "GoalPattern") -> bool:
        """
        Checks if this pattern (as a rule conclusion) can satisfy the given goal.
            - Variables in the conclusion (e.g., Var('X')) can match anything.
            - Concrete terms in the conclusion (e.g., Class('Person')) must match the goal.
        """

        if self.predicate != goal.predicate:
            return False

        # Prolog analogy: check if the terms can be unified
        def term_matches(rule_term: Term, goal_term: Term) -> bool:
            if isinstance(rule_term, Var):
                return True
            if isinstance(goal_term, Var):
                return True

            return rule_term == goal_term

        return term_matches(self.predicate, goal.predicate) and term_matches(
            self.object, goal.object
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
    conclusion: GoalPattern
    premises: List[GoalPattern]

    def __repr__(self):
        return f"Rule(name={self.name}, conc={self.conclusion}, prem={self.premises})"


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
