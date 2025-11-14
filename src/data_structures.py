"""
DESCRIPTION:

    Data structures for
        - Knowledge Graph representation
        - Rules and Constraints
        - Proofs and Backward Chaining

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
    """

    individual: Individual
    cls: Class
    is_member: bool
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
        return hash((self.individual.name, "rdf:type", self.cls.name))

    def to_atom(self) -> "Atom":
        """Converts this fact to a ground Atom."""
        return Atom(self.individual, RDF.type, self.cls)

    def __repr__(self) -> str:
        prefix = "" if self.is_member else "~"
        return f"<{self.individual}, {prefix}memberOf, {self.cls}>"


@dataclass
class Triple:
    """
    Represents a relational triple (subject, predicate, object).
    """

    subject: Individual
    predicate: Relation
    object: Individual
    positive: bool
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
        return hash((self.subject.name, self.predicate.name, self.object.name))

    def to_atom(self) -> "Atom":
        """Converts this fact to a ground Atom."""
        return Atom(self.subject, self.predicate, self.object)

    def __repr__(self) -> str:
        prefix = "" if self.positive else "~"
        return f"<{self.subject}, {prefix}{self.predicate}, {self.object}>"


@dataclass
class AttributeTriple:
    """
    Represents an attribute triple (subject, predicate, value).
    """

    subject: Individual
    predicate: Attribute
    value: LiteralValue
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
        return hash((self.subject.name, self.predicate.name, self.value))

    def to_atom(self) -> "Atom":
        """Converts this fact to a ground Atom."""
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

        # Print base vs inferred statistics
        base_triples = sum(1 for t in self.triples if t.is_base_fact)
        inferred_triples = len(self.triples) - base_triples
        base_memberships = sum(1 for m in self.memberships if m.is_base_fact)
        inferred_memberships = len(self.memberships) - base_memberships

        print(f"  Base Facts:")
        print(f"    Triples: {base_triples}")
        print(f"    Memberships: {base_memberships}")
        print(f"  Inferred Facts:")
        print(f"    Triples: {inferred_triples}")
        print(f"    Memberships: {inferred_memberships}\n")

        print("Triples:")
        for triple in self.triples:
            proof_info = (
                f" [base]"
                if triple.is_base_fact
                else f" [inferred, {len(triple.proofs)} proofs]"
            )
            print(f"    {triple}{proof_info}")
        print("Memberships:")
        for membership in self.memberships:
            proof_info = (
                f" [base]"
                if membership.is_base_fact
                else f" [inferred, {len(membership.proofs)} proofs]"
            )
            print(f"    {membership}{proof_info}")
        print("Attribute Triples:")
        for attr_triple in self.attribute_triples:
            proof_info = (
                f" [base]"
                if attr_triple.is_base_fact
                else f" [inferred, {len(attr_triple.proofs)} proofs]"
            )
            print(f"    {attr_triple}{proof_info}")


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


@dataclass(frozen=True)
class Var:
    """Represents a variable in a rule, e.g., 'X' or 'Y'."""

    name: str

    def __repr__(self):
        return f"{self.name}"


@dataclass(frozen=True)
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
        Checks if the atom is ground (no variables).
        """
        return not (
            isinstance(self.subject, Var)
            or isinstance(self.predicate, Var)
            or isinstance(self.object, Var)
        )

    def substitute(self, substitution: Dict[Var, Term]) -> "Atom":
        """
        Applies a variable substitution to this atom.
        """
        return Atom(
            subject=substitution.get(self.subject, self.subject),
            predicate=substitution.get(self.predicate, self.predicate),
            object=substitution.get(self.object, self.object),
        )

    def __repr__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


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
        if not self.premises:
            return f"-> {self.conclusion}  ({self.name})"
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
        head_type_term = None
        if head_pred == RDF.type and isinstance(head.object, (Class, Var)):
            head_type_term = head.object

        for premise in self.premises:
            # 1. Check predicate match
            if premise.predicate == head_pred:
                # 2. If predicate is rdf:type, check class match
                if head_type_term is not None:
                    if premise.object == head_type_term:
                        return True
                # 3. If predicate is not rdf:type, just predicate match is enough
                else:
                    return True
        return False


@dataclass
class Constraint:
    """
    Represents a constraint that must not be violated.
    E.g., owl:disjointWith(ClassA, ClassB)
    """

    name: str
    constraint_type: URIRef
    terms: List[Term]

    def __repr__(self):
        return f"Constraint(name={self.name}, type={self.constraint_type}, terms={self.terms})"


@dataclass(frozen=True)
class Proof:
    """
    Represents a proof tree for a single GROUND goal (Atom).

    A proof is either for
        - a base fact (leaf in the proof tree, rule=None)
        - a derived fact (node in the proof tree, rule=ExecutableRule, sub_proofs=[Proof, ...])
    """

    goal: Atom
    rule: Optional[ExecutableRule] = None
    sub_proofs: Tuple["Proof", ...] = field(default_factory=tuple)
    recursive_use_counts: frozenset[Tuple[str, int]] = field(default_factory=frozenset)

    def __post_init__(self):
        if self.rule is None and self.sub_proofs:
            raise ValueError("Base fact proof (rule=None) cannot have sub-proofs.")

        if self.rule is not None and len(self.sub_proofs) != len(self.rule.premises):
            raise ValueError(
                f"Proof for rule '{self.rule.name}' must have "
                f"{len(self.rule.premises)} sub-proofs, but "
                f"{len(self.sub_proofs)} were given."
            )

        if not self.goal.is_ground():
            raise ValueError(f"Proof goal '{self.goal}' must be a ground atom.")

    def is_base_fact(self) -> bool:
        """Checks if this proof represents a base fact (leaf)."""
        return self.rule is None

    def get_base_facts(self) -> Set[Atom]:
        """Returns the set of all base facts (leaves) this proof depends on."""
        if self.rule is None:
            return {self.goal}

        base_facts: Set[Atom] = set()
        for sp in self.sub_proofs:
            base_facts.update(sp.get_base_facts())
        return base_facts

    def get_all_atoms(self) -> Set[Atom]:
        """Returns all atoms (base and derived) in this proof tree."""
        atoms = {self.goal}
        for sp in self.sub_proofs:
            atoms.update(sp.get_all_atoms())
        return atoms

    def get_recursion_depth(self, rule: ExecutableRule) -> int:
        """Gets the number of times the given recursive rule was used in this proof path."""
        for name, count in self.recursive_use_counts:
            if name == rule.name:
                return count
        return 0

    def get_max_recursion_depth(self) -> int:
        """Gets the maximum depth of any recursive rule in this proof."""
        if not self.recursive_use_counts:
            return 0
        return max(count for _, count in self.recursive_use_counts)

    def depth(self) -> int:
        """Returns the depth of this proof tree."""
        if self.rule is None:
            return 0
        if not self.sub_proofs:
            return 1
        return 1 + max(sp.depth() for sp in self.sub_proofs)

    def __repr__(self) -> str:
        if self.is_base_fact():
            return f"BaseProof({self.goal})"
        return f"DerivedProof({self.goal} via {self.rule.name})"

    @staticmethod
    def create_base_proof(atom: Atom) -> "Proof":
        """Creates a proof for a base fact (a leaf)."""
        if not atom.is_ground():
            raise ValueError("Base fact proof must be for a ground atom.")
        return Proof(goal=atom, rule=None, sub_proofs=tuple())

    @staticmethod
    def create_derived_proof(
        goal: Atom, rule: ExecutableRule, sub_proofs: List["Proof"]
    ) -> "Proof":
        """Creates a proof for a derived fact (a node), tracking recursion."""
        if not goal.is_ground():
            raise ValueError("Derived proof goal must be a ground atom.")

        # Combine recursive_use_counts from sub-proofs
        new_counts: Dict[str, int] = {}
        for sp in sub_proofs:
            for name, count in sp.recursive_use_counts:
                new_counts[name] = max(new_counts.get(name, 0), count)

        # Update this rule's count if it's recursive
        if rule.is_recursive():
            name = rule.name
            new_counts[name] = new_counts.get(name, 0) + 1

        return Proof(
            goal=goal,
            rule=rule,
            sub_proofs=tuple(sub_proofs),
            recursive_use_counts=frozenset(new_counts.items()),
        )
