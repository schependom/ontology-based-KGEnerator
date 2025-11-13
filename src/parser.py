"""
DESCRIPTION

    Parses an OWL ontology file (.ttl) and extracts classes, relations,
    attributes, executable rules, and constraints for use in backward chaining.

WORKFLOW

    1. Load the ontology using rdflib.
    2. Identify and create Class, Relation, and Attribute entities.
    3. Translate OWL/RDFS axioms into ExecutableRule and Constraint objects.

AUTHOR

    Vincent Van Schependom
"""

from rdflib.namespace import RDF, RDFS, OWL, XSD
from rdflib.term import BNode, URIRef, Literal
import rdflib

from typing import List, Union, Set, Dict, Optional

from data_structures import (
    Individual,
    Class,
    Relation,
    Attribute,
    Var,
    Term,
    Atom,
    ExecutableRule,
    Constraint,
)


# ---------------------------------------------------------------------------- #
#                                    HELPER                                    #
# ---------------------------------------------------------------------------- #


def remove_uri_prefix(uri: str) -> str:
    """Removes common URI prefixes for better readability."""
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.split("/")[-1]
    return uri


# ---------------------------------------------------------------------------- #
#                                 ACTUAL PARSER                                #
# ---------------------------------------------------------------------------- #


class OntologyParser:
    def __init__(
        self,
        filepath: str,
        filetype: Optional[str] = "turtle",
        seed: Optional[int] = None,
    ):
        """
        Parses an OWL ontology and extracts classes, relations, rules, and constraints.

        Args:
            filepath (str):             Path to the ontology file.
            filetype (str, optional):   Format of the ontology file (default: "turtle").
        """

        self.graph = rdflib.Graph()
        self.graph.parse(filepath, format=filetype)

        # Entity storage
        self.classes: Dict[str, Class] = {}  # name -> Class
        self.relations: Dict[str, Relation] = {}  # name -> Relation
        self.attributes: Dict[str, Attribute] = {}  # name -> Attribute

        # Entity sets for quick lookup by URI
        self._class_uris: Set[str] = set()
        self._relation_uris: Set[str] = set()
        self._attribute_uris: Set[str] = set()

        # Output lists
        self.rules: List[ExecutableRule] = []
        self.constraints: List[Constraint] = []

        # Helper map for constraint propagation
        self.sub_property_map: Dict[str, Set[Relation]] = {}

        self._parse_schema()
        self._parse_rules_and_constraints()

    # ---------------------------------------------------------------------------- #
    #                                 SCHEMA PARSING                               #
    # ---------------------------------------------------------------------------- #

    def _get_or_create_class(self, uri: URIRef) -> Optional[Class]:
        """Gets or creates a Class from a URI."""
        if isinstance(uri, BNode) or isinstance(uri, Literal):
            return None

        uri_str = str(uri)
        name = remove_uri_prefix(uri_str)

        if name in self.classes:
            return self.classes[name]

        # Avoid creating a class if it's already a property
        if name in self.relations or name in self.attributes:
            return None

        # Create new class
        index = len(self.classes)
        cls = Class(index, name)
        self.classes[name] = cls
        self._class_uris.add(uri_str)
        return cls

    def _get_or_create_relation(self, uri: URIRef) -> Optional[Relation]:
        """Gets or creates a Relation (ObjectProperty) from a URI."""
        if isinstance(uri, BNode) or isinstance(uri, Literal):
            return None

        uri_str = str(uri)
        name = remove_uri_prefix(uri_str)

        if name in self.relations:
            return self.relations[name]

        # Avoid creating a relation if it's a class or attribute
        if name in self.classes or name in self.attributes:
            return None

        index = len(self.relations)
        rel = Relation(index, name)
        self.relations[name] = rel
        self._relation_uris.add(uri_str)
        return rel

    def _get_or_create_attribute(self, uri: URIRef) -> Optional[Attribute]:
        """Gets or creates an Attribute (DatatypeProperty) from a URI."""
        if isinstance(uri, BNode) or isinstance(uri, Literal):
            return None

        uri_str = str(uri)
        name = remove_uri_prefix(uri_str)

        if name in self.attributes:
            return self.attributes[name]

        # Avoid creating an attribute if it's a class or relation
        if name in self.classes or name in self.relations:
            return None

        index = len(self.attributes)
        attr = Attribute(index, name)
        self.attributes[name] = attr
        self._attribute_uris.add(uri_str)
        return attr

    def _parse_schema(self) -> None:
        """
        Parses the ontology schema to discover all Classes, Relations, and Attributes.
        """

        # Find explicit declarations
        for s in self.graph.subjects(RDF.type, OWL.Class):
            self._get_or_create_class(s)

        for s in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            self._get_or_create_relation(s)

        for s in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            self._get_or_create_attribute(s)

        #  ind property types that imply ObjectProperty
        prop_types = [
            OWL.SymmetricProperty,
            OWL.TransitiveProperty,
            OWL.InverseFunctionalProperty,
            OWL.IrreflexiveProperty,
        ]
        for prop_type in prop_types:
            for s in self.graph.subjects(RDF.type, prop_type):
                self._get_or_create_relation(s)

        #  Handle FunctionalProperty (can be Object or Datatype)
        for s in self.graph.subjects(RDF.type, OWL.FunctionalProperty):
            # If already typed, we're good. If not, we must deduce.
            uri_str = str(s)
            if (
                uri_str not in self._relation_uris
                and uri_str not in self._attribute_uris
            ):
                # Deduce from range: if range is a datatype, it's an Attribute.
                is_attr = False
                for o_range in self.graph.objects(s, RDFS.range):
                    if (o_range, RDF.type, RDFS.Datatype) in self.graph or str(
                        o_range
                    ).startswith(str(XSD)):
                        is_attr = True
                        break

                if is_attr:
                    self._get_or_create_attribute(s)
                else:
                    self._get_or_create_relation(s)  # Default to relation

        # Find implicit entities from axiom usage
        # We must iterate all triples, but this is safer.

        # rdfs:subClassOf
        for s, o in self.graph.subject_objects(RDFS.subClassOf):
            self._get_or_create_class(s)
            self._get_or_create_class(o)

        # owl:disjointWith
        for s, o in self.graph.subject_objects(OWL.disjointWith):
            self._get_or_create_class(s)
            self._get_or_create_class(o)

        # rdfs:domain
        for p, c in self.graph.subject_objects(RDFS.domain):
            self._get_or_create_class(c)
            # We don't know p's type, but _parse_rules will need it.
            # Let's try to find it.
            if str(p) not in self._relation_uris and str(p) not in self._attribute_uris:
                self._deduce_property_type(p)

        # rdfs:range
        for p, c in self.graph.subject_objects(RDFS.range):
            # If range is a datatype, p is an Attribute
            if (c, RDF.type, RDFS.Datatype) in self.graph or str(c).startswith(
                str(XSD)
            ):
                self._get_or_create_attribute(p)
            else:
                # Otherwise, p is a Relation and c is a Class
                self._get_or_create_relation(p)
                self._get_or_create_class(c)

        # rdfs:subPropertyOf
        for s, o in self.graph.subject_objects(RDFS.subPropertyOf):
            # Assume same type for s and o. Deduce if unknown.
            if str(s) in self._relation_uris or str(o) in self._relation_uris:
                self._get_or_create_relation(s)
                self._get_or_create_relation(o)
            elif str(s) in self._attribute_uris or str(o) in self._attribute_uris:
                self._get_or_create_attribute(s)
                self._get_or_create_attribute(o)
            else:
                # Both unknown. Deduce from range.
                if self._deduce_property_type(s) == "attribute":
                    self._get_or_create_attribute(s)
                    self._get_or_create_attribute(o)
                else:
                    self._get_or_create_relation(s)
                    self._get_or_create_relation(o)

        # owl:inverseOf
        for s, o in self.graph.subject_objects(OWL.inverseOf):
            self._get_or_create_relation(s)
            self._get_or_create_relation(o)

        # someValuesFrom / allValuesFrom
        for s in self.graph.subjects(OWL.onProperty, None):
            if (s, RDF.type, OWL.Restriction) in self.graph:
                prop = next(self.graph.objects(s, OWL.onProperty), None)
                val_class = next(
                    self.graph.objects(s, OWL.someValuesFrom), None
                ) or next(self.graph.objects(s, OWL.allValuesFrom), None)

                if prop and val_class:
                    # someValuesFrom can be on Datatype or Object properties
                    if (val_class, RDF.type, RDFS.Datatype) in self.graph or str(
                        val_class
                    ).startswith(str(XSD)):
                        self._get_or_create_attribute(prop)
                    else:
                        self._get_or_create_relation(prop)
                        self._get_or_create_class(val_class)

        print("--- Ontology Parsing Results ---")
        print(f"Total classes found: {len(self.classes)}")
        print(f"Total relations found: {len(self.relations)}")
        print(f"Total attributes found: {len(self.attributes)}")

    def _deduce_property_type(self, p: URIRef) -> str:
        """Helper to deduce if a property is Object or Datatype. Returns 'relation' or 'attribute'."""
        if str(p) in self._relation_uris:
            return "relation"
        if str(p) in self._attribute_uris:
            return "attribute"

        is_attr = False
        for o_range in self.graph.objects(p, RDFS.range):
            if (o_range, RDF.type, RDFS.Datatype) in self.graph or str(
                o_range
            ).startswith(str(XSD)):
                is_attr = True
                break

        if is_attr:
            self._get_or_create_attribute(p)
            return "attribute"
        else:
            self._get_or_create_relation(p)
            return "relation"

    # ---------------------------------------------------------------------------- #
    #                               HELPER FUNCTIONS                               #
    # ---------------------------------------------------------------------------- #

    def get_class(self, term: Union[URIRef, str]) -> Optional[Class]:
        """Retrieve a class by its URI or name."""
        name = remove_uri_prefix(str(term))
        return self.classes.get(name)

    def get_object_property(self, term: Union[URIRef, str]) -> Optional[Relation]:
        """Retrieve an object property (relation) by its URI or name."""
        name = remove_uri_prefix(str(term))
        return self.relations.get(name)

    def get_attribute_property(self, term: Union[URIRef, str]) -> Optional[Attribute]:
        """Retrieve an attribute property by its URI or name."""
        name = remove_uri_prefix(str(term))
        return self.attributes.get(name)

    # ---------------------------------------------------------------------------- #
    #                          PARSE RULES AND CONSTRAINTS                         #
    # ---------------------------------------------------------------------------- #

    def _parse_rules_and_constraints(self) -> None:
        """
        Translates OWL/RDFS axioms into executable rules and constraints
        using the pre-parsed schema entities.
        """

        # --- rdfs:subClassOf ---
        # <A, rdfs:subClassOf, B>  =>  (X, type, A) -> (X, type, B)
        for s, o in self.graph.subject_objects(RDFS.subClassOf):
            if isinstance(s, BNode) or isinstance(o, BNode):
                self._parse_complex_subclass(s, o)
            else:
                class_s = self.get_class(s)
                class_o = self.get_class(o)
                if class_s and class_o:
                    rule = ExecutableRule(
                        name=f"subClassOf_{class_s.name}_{class_o.name}",
                        conclusion=Atom(Var("X"), RDF.type, class_o),
                        premises=[Atom(Var("X"), RDF.type, class_s)],
                    )
                    self.rules.append(rule)

        # --- rdfs:subPropertyOf ---
        # <P1, rdfs:subPropertyOf, P2>  =>  (X, P1, Y) -> (X, P2, Y)
        for s, o in self.graph.subject_objects(RDFS.subPropertyOf):
            rel_s = self.get_object_property(s)
            rel_o = self.get_object_property(o)
            if rel_s and rel_o:
                rule = ExecutableRule(
                    name=f"subPropertyOf_{rel_s.name}_{rel_o.name}",
                    conclusion=Atom(Var("X"), rel_o, Var("Y")),
                    premises=[Atom(Var("X"), rel_s, Var("Y"))],
                )
                self.rules.append(rule)
                # Populate sub_property_map
                if rel_o.name not in self.sub_property_map:
                    self.sub_property_map[rel_o.name] = set()
                self.sub_property_map[rel_o.name].add(rel_s)

            attr_s = self.get_attribute_property(s)
            attr_o = self.get_attribute_property(o)
            if attr_s and attr_o:
                rule = ExecutableRule(
                    name=f"subPropertyOf_{attr_s.name}_{attr_o.name}",
                    conclusion=Atom(Var("X"), attr_o, Var("Value")),
                    premises=[Atom(Var("X"), attr_s, Var("Value"))],
                )
                self.rules.append(rule)

        # --- rdfs:domain ---
        # <P, rdfs:domain, C>  =>  (X, P, Y) -> (X, type, C)
        # <A, rdfs:domain, C>  =>  (X, A, V) -> (X, type, C)
        for s, o in self.graph.subject_objects(RDFS.domain):
            class_o = self.get_class(o)
            if not class_o:
                continue

            rel_s = self.get_object_property(s)
            if rel_s:
                rule = ExecutableRule(
                    name=f"domain_{rel_s.name}_{class_o.name}",
                    conclusion=Atom(Var("X"), RDF.type, class_o),
                    premises=[Atom(Var("X"), rel_s, Var("Y"))],
                )
                self.rules.append(rule)

            attr_s = self.get_attribute_property(s)
            if attr_s:
                rule = ExecutableRule(
                    name=f"domain_{attr_s.name}_{class_o.name}",
                    conclusion=Atom(Var("X"), RDF.type, class_o),
                    premises=[Atom(Var("X"), attr_s, Var("Value"))],
                )
                self.rules.append(rule)

        # --- rdfs:range ---
        # <P, rdfs:range, C>  =>  (X, P, Y) -> (Y, type, C) (ObjectProperty)
        # <A, rdfs:range, D>  =>  (X, A, V) -> V is of type D (DatatypeProperty)
        for s, o in self.graph.subject_objects(RDFS.range):
            rel_s = self.get_object_property(s)
            class_o = self.get_class(o)
            if rel_s and class_o:
                rule = ExecutableRule(
                    name=f"range_{rel_s.name}_{class_o.name}",
                    conclusion=Atom(Var("Y"), RDF.type, class_o),
                    premises=[Atom(Var("X"), rel_s, Var("Y"))],
                )
                self.rules.append(rule)

            attr_s = self.get_attribute_property(s)
            if attr_s:
                # This is a literal type constraint
                self.constraints.append(
                    Constraint(
                        name=f"range_{attr_s.name}_{remove_uri_prefix(str(o))}",
                        constraint_type=RDFS.range,
                        terms=[
                            attr_s,
                            o,
                        ],  # Store the attribute and the XSD type (e.g., XSD.string)
                    )
                )

        # --- owl:inverseOf ---
        # <P1, owl:inverseOf, P2> => (X, P1, Y) -> (Y, P2, X)
        #                        and (Y, P2, X) -> (X, P1, Y)
        for s, o in self.graph.subject_objects(OWL.inverseOf):
            rel_s = self.get_object_property(s)
            rel_o = self.get_object_property(o)
            if rel_s and rel_o:
                # Rule 1
                self.rules.append(
                    ExecutableRule(
                        name=f"inverseOf_{rel_s.name}_{rel_o.name}",
                        conclusion=Atom(Var("Y"), rel_o, Var("X")),
                        premises=[Atom(Var("X"), rel_s, Var("Y"))],
                    )
                )
                # Rule 2
                self.rules.append(
                    ExecutableRule(
                        name=f"inverseOf_{rel_o.name}_{rel_s.name}",
                        conclusion=Atom(Var("X"), rel_s, Var("Y")),
                        premises=[Atom(Var("Y"), rel_o, Var("X"))],
                    )
                )

        # --- owl:SymmetricProperty ---
        # <P, rdf:type, owl:SymmetricProperty> => (X, P, Y) -> (Y, P, X)
        for s in self.graph.subjects(RDF.type, OWL.SymmetricProperty):
            rel_s = self.get_object_property(s)
            if rel_s:
                self.rules.append(
                    ExecutableRule(
                        name=f"symmetric_{rel_s.name}",
                        conclusion=Atom(Var("Y"), rel_s, Var("X")),
                        premises=[Atom(Var("X"), rel_s, Var("Y"))],
                    )
                )

        # --- owl:TransitiveProperty ---
        # <P, rdf:type, owl:TransitiveProperty> => (X, P, Y), (Y, P, Z) -> (X, P, Z)
        for s in self.graph.subjects(RDF.type, OWL.TransitiveProperty):
            rel_s = self.get_object_property(s)
            if rel_s:
                self.rules.append(
                    ExecutableRule(
                        name=f"transitive_{rel_s.name}",
                        conclusion=Atom(Var("X"), rel_s, Var("Z")),
                        premises=[
                            Atom(Var("X"), rel_s, Var("Y")),
                            Atom(Var("Y"), rel_s, Var("Z")),
                        ],
                    )
                )

        # --- owl:disjointWith ---
        # <A, owl:disjointWith, B> => Constraint
        for s, o in self.graph.subject_objects(OWL.disjointWith):
            class_s = self.get_class(s)
            class_o = self.get_class(o)
            if class_s and class_o:
                self.constraints.append(
                    Constraint(
                        name=f"disjointWith_{class_s.name}_{class_o.name}",
                        constraint_type=OWL.disjointWith,
                        terms=[class_s, class_o],
                    )
                )

        # --- owl:FunctionalProperty ---
        for s in self.graph.subjects(RDF.type, OWL.FunctionalProperty):
            rel_s = self.get_object_property(s)
            if rel_s:
                self.constraints.append(
                    Constraint(
                        name=f"functional_{rel_s.name}",
                        constraint_type=OWL.FunctionalProperty,
                        terms=[rel_s],
                    )
                )

            attr_s = self.get_attribute_property(s)
            if attr_s:
                self.constraints.append(
                    Constraint(
                        name=f"functional_{attr_s.name}",
                        constraint_type=OWL.FunctionalProperty,
                        terms=[attr_s],
                    )
                )

        # --- owl:IrreflexiveProperty ---
        for s in self.graph.subjects(RDF.type, OWL.IrreflexiveProperty):
            prop = self.get_object_property(s)  # e.g., hasParent
            if prop:
                # Add constraint for the property itself
                self.constraints.append(
                    Constraint(
                        name=f"irreflexive_{prop.name}",
                        constraint_type=OWL.IrreflexiveProperty,
                        terms=[prop],
                    )
                )
                # Add constraint for all sub-properties
                if prop.name in self.sub_property_map:
                    for sub_prop in self.sub_property_map[prop.name]:
                        self.constraints.append(
                            Constraint(
                                name=f"irreflexive_{sub_prop.name}_(inherited)",
                                constraint_type=OWL.IrreflexiveProperty,
                                terms=[sub_prop],
                            )
                        )

        print(f"Total rules parsed: {len(self.rules)}")
        print(f"Total constraints parsed: {len(self.constraints)}")
        print("----------------------------------")

    # ---------------------------------------------------------------------------- #
    #                                    COMPLEX                                   #
    # ---------------------------------------------------------------------------- #

    def _parse_complex_subclass(self, s: Term, o: Term) -> None:
        """
        Parses complex class expressions (e.g., restrictions) in subclass axioms.
        """

        # Example for owl:someValuesFrom
        # [a owl:Restriction; owl:onProperty P; owl:someValuesFrom C] rdfs:subClassOf ClassA
        # This means: (X, P, Y), (Y, type, C) -> (X, type, ClassA)
        if isinstance(s, BNode):
            try:
                # Check for: (s, rdf:type, owl:Restriction)
                if (s, RDF.type, OWL.Restriction) not in self.graph:
                    return

                prop = next(self.graph.objects(s, OWL.onProperty))
                some_class = next(self.graph.objects(s, OWL.someValuesFrom))

                rel_p = self.get_object_property(prop)  # Can only be ObjectProperty
                class_c = self.get_class(some_class)
                class_a = self.get_class(o)  # The conclusion

                if rel_p and class_c and class_a:
                    rule = ExecutableRule(
                        name=f"someValuesFrom_{rel_p.name}_{class_c.name}_{class_a.name}",
                        conclusion=Atom(Var("X"), RDF.type, class_a),
                        premises=[
                            Atom(Var("X"), rel_p, Var("Y")),
                            Atom(Var("Y"), RDF.type, class_c),
                        ],
                    )
                    self.rules.append(rule)

            except StopIteration:
                pass

        # ... other complex axioms like allValuesFrom, hasValue, etc. can be added here
