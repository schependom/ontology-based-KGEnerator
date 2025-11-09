from rdflib.namespace import RDF, RDFS, OWL, XSD
from rdflib.term import BNode, URIRef, Literal
import rdflib

from dataclasses import dataclass
from typing import List, Union, Set, Dict, Optional, Any
import sys

from data_structures import (
    Individual,
    Class,
    Relation,
    Attribute,
    Var,
    Term,
    GoalPattern,
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

        self.classes: Dict[str, Class] = {}  # <s, isMemberOf, o>
        self.relations: Dict[str, Relation] = {}  # <s, p, o> where o is an Individual
        self.attributes: Dict[str, Attribute] = {}  # <s, p, l> where l is a literal

        self.rules: List[ExecutableRule] = []
        self.constraints: List[Constraint] = []

        # Propagate constraints
        self.sub_property_map: Dict[str, Set[Relation]] = {}

        self._parse_schema()
        self._parse_rules_and_constraints()

    # ---------------------------------------------------------------------------- #
    #                                 PARSE SCHEMA                                 #
    # ---------------------------------------------------------------------------- #

    def _parse_schema(self) -> None:
        """
        Parses the ontology schema to extract
            - Class objects
            - Relation objects
            - Attribute objects
        """

        # For object creation, we need unique indices
        cls_idx = 0
        rel_idx = 0
        attr_idx = 0

        # Prevent duplicates
        cls_set: Set[str] = set()
        rel_set: Set[str] = set()
        attr_set: Set[str] = set()

        # ------------ Find all EXPLICITLY declared relations & attributes ----------- #

        # ObjectProperties, like <x, hasParent, y>
        for s in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            if not isinstance(s, BNode) and str(s) not in rel_set:
                name = remove_uri_prefix(str(s))
                self.relations[name] = Relation(rel_idx, name)
                rel_idx += 1
                rel_set.add(str(s))

        # DatatypeProperties, like <x, hasAge, "30">, where "30" is a literal
        for s in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            if not isinstance(s, BNode) and str(s) not in attr_set:
                name = remove_uri_prefix(str(s))
                self.attributes[name] = Attribute(attr_idx, name)
                attr_idx += 1
                attr_set.add(str(s))

        # Symmetric, Transitive, and Inverse properties
        prop_types = [
            OWL.SymmetricProperty,
            OWL.TransitiveProperty,
            OWL.InverseFunctionalProperty,
        ]
        for prop_type in prop_types:
            for s in self.graph.subjects(RDF.type, prop_type):
                if not isinstance(s, BNode) and str(s) not in rel_set:
                    name = remove_uri_prefix(str(s))
                    if name not in self.relations:
                        self.relations[name] = Relation(rel_idx, name)
                        rel_idx += 1
                        rel_set.add(str(s))

        # Functional (Datatype or Object) properties
        #
        # A functional property is a property that can have only one value for a given subject.
        # If <a, FunctionalProperty, b> -> <a, FunctionalProperty, c> is impossible if b != c.
        #
        for s in self.graph.subjects(RDF.type, OWL.FunctionalProperty):
            if not isinstance(s, BNode):
                #
                name = remove_uri_prefix(str(s))
                s_str = str(s)

                # A functional property can be either an
                #   -   ObjectProperty (object=Individual)
                #   -   DatatypeProperty (object=Literal)

                # If we didn't encounter it yet as either, we need to determine which one it is.
                if s_str not in rel_set and s_str not in attr_set:
                    #
                    # It is an ObjectProperty
                    if (s, RDF.type, OWL.ObjectProperty) in self.graph:
                        if name not in self.relations:
                            self.relations[name] = Relation(rel_idx, name)
                            rel_idx += 1
                            rel_set.add(s_str)

                    # It is a DatatypeProperty
                    elif (s, RDF.type, OWL.DatatypeProperty) in self.graph:
                        if name not in self.attributes:
                            self.attributes[name] = Attribute(attr_idx, name)
                            attr_idx += 1
                            attr_set.add(s_str)

                    # It's functional but not explicitly typed.
                    else:
                        print(
                            f"Warning: Functional property '{name}' is not explicitly typed.",
                            file=sys.stderr,
                        )
                        pass

        # ------------------- Find all EXPLICITLY declared classes ------------------- #

        for s in self.graph.subjects(RDF.type, OWL.Class):
            if not isinstance(s, BNode) and str(s) not in cls_set:
                name = remove_uri_prefix(str(s))
                if name not in self.relations and name not in self.attributes:
                    self.classes[name] = Class(cls_idx, name)
                    cls_idx += 1
                    cls_set.add(str(s))

        # ------------ Find all IMPLICITLY defined relations & attributes ------------ #

        # Find properties from domain/range/inverseof/subPropertyOf
        rel_axioms = [RDFS.domain, RDFS.range, OWL.inverseOf, RDFS.subPropertyOf]

        for axiom in rel_axioms:
            #
            # For domain and range, 'o' is a class or datatype
            # For inverseOf and subPropertyOf, 'o' is also a property
            for p, o in self.graph.subject_objects(axiom):
                p_str = str(p)
                name = remove_uri_prefix(p_str)

                # We want to check if p is not already known, because
                # if it is, we don't need to re-check its type.
                if (
                    not isinstance(p, BNode)
                    and p_str not in rel_set
                    and p_str not in attr_set
                ):
                    # We don't yet know if p is an Attribute or a Relation.
                    # So we determine it now based on its range.
                    #   ->  If the range is a datatype, it's an attribute.
                    #   ->  Otherwise, it's a relation.
                    #
                    is_attr = False

                    # Check all ranges for this property (p)
                    for o_range in self.graph.objects(p, RDFS.range):
                        # If the range is a datatype, it's an attribute
                        if (o_range, RDF.type, RDFS.Datatype) in self.graph or str(
                            o_range
                        ).startswith(str(XSD)):
                            is_attr = True
                            break

                    # Now we know if it's an attribute or relation!
                    # Let's create the appropriate object.
                    if is_attr:
                        # Range
                        if name not in self.attributes:
                            self.attributes[name] = Attribute(attr_idx, name)
                            attr_idx += 1
                            attr_set.add(p_str)
                    else:
                        # Inverse, SubProperty
                        if name not in self.relations:
                            self.relations[name] = Relation(rel_idx, name)
                            rel_idx += 1
                            rel_set.add(p_str)

                # For inverseOf and subPropertyOf, the subject is a property and the object is also a property,
                # so we immediately create the inverse property if not present.
                if axiom in [OWL.inverseOf, RDFS.subPropertyOf]:
                    # For inverseOf/subPropertyOf, 'o' is also a property,
                    # so we call it 'inverse_p' here for clarity.
                    inverse_p_str = str(o)
                    if not isinstance(o, BNode) and inverse_p_str not in rel_set:
                        name = remove_uri_prefix(inverse_p_str)
                        # Assuming these axioms only apply to ObjectProperty
                        if name not in self.relations:
                            self.relations[name] = Relation(rel_idx, name)
                            rel_idx += 1
                            rel_set.add(inverse_p_str)

        # -------------------- Find all IMPLICITLY defined classes ------------------- #

        #### From domain/range objects ###
        rel_class_axioms = [RDFS.domain, RDFS.range]

        for axiom in rel_class_axioms:
            # domain: <d, rdfs:domain, C>
            # range:  <r, rdfs:range, C>
            # So, d and r are properties, C is a class
            for property, cls in self.graph.subject_objects(axiom):
                cls_str = str(cls)

                # Check if cls is not a literal
                # and if it isn't (and thus is a class) it is not already known
                if not isinstance(cls, (BNode, Literal)) and cls_str not in cls_set:
                    cls_name = remove_uri_prefix(cls_str)
                    if (
                        cls_name
                        not in self.relations  # why? because a property can't also be a class
                        and cls_name
                        not in self.attributes  # why? because an attribute can't also be a class
                        and cls_name not in self.classes  # to avoid duplicates
                    ):
                        self.classes[cls_name] = Class(cls_idx, cls_name)
                        cls_idx += 1
                        cls_set.add(cls_str)

        #### From subClassOf/disjointWith subjects/objects ####

        class_axioms = [RDFS.subClassOf, OWL.disjointWith]

        for axiom in class_axioms:
            for c1, c2 in self.graph.subject_objects(axiom):
                c1_str = str(c1)
                c2_str = str(c2)

                # Not yet encountered c1
                if not isinstance(c1, (BNode, Literal)) and c1_str not in cls_set:
                    name = remove_uri_prefix(c1_str)
                    if (
                        name not in self.relations
                        and name not in self.attributes
                        and name not in self.classes
                    ):
                        self.classes[name] = Class(cls_idx, name)
                        cls_idx += 1
                        cls_set.add(c1_str)

                # Not yet encountered c2
                if not isinstance(c2, (BNode, Literal)) and c2_str not in cls_set:
                    name = remove_uri_prefix(c2_str)
                    if (
                        name not in self.relations
                        and name not in self.attributes
                        and name not in self.classes
                    ):
                        self.classes[name] = Class(cls_idx, name)
                        cls_idx += 1
                        cls_set.add(c2_str)

        print("--- Ontology Parsing Results ---")
        print(f"Total classes found: {len(self.classes)}")
        print(f"Total relations found: {len(self.relations)}")
        print(f"Total attributes found: {len(self.attributes)}")  # --- NEW ---

    # ---------------------------------------------------------------------------- #
    #                               HELPER FUNCTIONS                               #
    # ---------------------------------------------------------------------------- #

    def get_class(self, term: Union[URIRef, str]) -> Optional[Class]:
        """
        Retrieve a class by its URI or name.
        """
        name = remove_uri_prefix(str(term))
        return self.classes.get(name)

    def get_object_property(self, term: Union[URIRef, str]) -> Optional[Relation]:
        """
        Retrieve an object property (relation) by its URI or name.
        """
        name = remove_uri_prefix(str(term))
        return self.relations.get(name)

    def get_attribute_property(self, term: Union[URIRef, str]) -> Optional[Attribute]:
        """
        Retrieve an attribute property by its URI or name.
        """
        name = remove_uri_prefix(str(term))
        return self.attributes.get(name)

    # ---------------------------------------------------------------------------- #
    #                          PARSE RULES AND CONSTRAINTS                         #
    # ---------------------------------------------------------------------------- #

    def _parse_rules_and_constraints(self) -> None:
        """
        Translates OWL/RDFS axioms into executable rules and constraints.
        """

        # ------------------------------ SUBPROPERTY MAP ----------------------------- #

        # This map helps to propagate constraints from super-properties to sub-properties.
        # e.g., if hasMother is a subPropertyOf hasParent, and hasParent is irreflexive,
        # then hasMother should also be irreflexive.

        for s, o in self.graph.subject_objects(RDFS.subPropertyOf):
            # This can apply to both Object and Datatype properties
            # -> e.g. Object: hasMother subPropertyOf hasParent
            # -> e.g. Datatype: hasAgeInYears subPropertyOf hasAge
            rel_s = self.get_object_property(s)  # e.g., hasMother
            rel_o = self.get_object_property(o)  # e.g., hasParent
            if rel_s and rel_o:
                if rel_o.name not in self.sub_property_map:
                    self.sub_property_map[rel_o.name] = set()
                self.sub_property_map[rel_o.name].add(rel_s)

        # ------------------------ PARSE RULES AND CONSTRAINTS ----------------------- #

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
                        conclusion=GoalPattern(Var("X"), RDF.type, class_o),
                        premises=[GoalPattern(Var("X"), RDF.type, class_s)],
                    )
                    self.rules.append(rule)

        # --- rdfs:subPropertyOf ---
        # <P1, rdfs:subPropertyOf, P2>  =>  (X, P1, Y) -> (X, P2, Y)
        for s, o in self.graph.subject_objects(RDFS.subPropertyOf):
            # This can apply to both Object and Datatype properties
            rel_s = self.get_object_property(s)
            rel_o = self.get_object_property(o)
            if rel_s and rel_o:
                rule = ExecutableRule(
                    name=f"subPropertyOf_{rel_s.name}_{rel_o.name}",
                    conclusion=GoalPattern(Var("X"), rel_o, Var("Y")),
                    premises=[GoalPattern(Var("X"), rel_s, Var("Y"))],
                )
                self.rules.append(rule)

            attr_s = self.get_attribute_property(s)
            attr_o = self.get_attribute_property(o)
            if attr_s and attr_o:
                rule = ExecutableRule(
                    name=f"subPropertyOf_{attr_s.name}_{attr_o.name}",
                    conclusion=GoalPattern(Var("X"), attr_o, Var("Value")),
                    premises=[GoalPattern(Var("X"), attr_s, Var("Value"))],
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
                    conclusion=GoalPattern(Var("X"), RDF.type, class_o),
                    premises=[GoalPattern(Var("X"), rel_s, Var("Y"))],
                )
                self.rules.append(rule)

            attr_s = self.get_attribute_property(s)
            if attr_s:
                rule = ExecutableRule(
                    name=f"domain_{attr_s.name}_{class_o.name}",
                    conclusion=GoalPattern(Var("X"), RDF.type, class_o),
                    premises=[GoalPattern(Var("X"), attr_s, Var("Value"))],
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
                    conclusion=GoalPattern(Var("Y"), RDF.type, class_o),
                    premises=[GoalPattern(Var("X"), rel_s, Var("Y"))],
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
                        conclusion=GoalPattern(Var("Y"), rel_o, Var("X")),
                        premises=[GoalPattern(Var("X"), rel_s, Var("Y"))],
                    )
                )
                # Rule 2
                self.rules.append(
                    ExecutableRule(
                        name=f"inverseOf_{rel_o.name}_{rel_s.name}",
                        conclusion=GoalPattern(Var("X"), rel_s, Var("Y")),
                        premises=[GoalPattern(Var("Y"), rel_o, Var("X"))],
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
                        conclusion=GoalPattern(Var("Y"), rel_s, Var("X")),
                        premises=[GoalPattern(Var("X"), rel_s, Var("Y"))],
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
                        conclusion=GoalPattern(Var("X"), rel_s, Var("Z")),
                        premises=[
                            GoalPattern(Var("X"), rel_s, Var("Y")),
                            GoalPattern(Var("Y"), rel_s, Var("Z")),
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
        # This can apply to BOTH ObjectProperty and DatatypeProperty
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

            # Handle properties that are only functional
            if not rel_s and not attr_s:
                # We don't know what it is, but we need to track it
                # as some kind of property for the constraint.
                # Let's check its range.
                is_attr = False
                for o_range in self.graph.objects(s, RDFS.range):
                    if (o_range, RDF.type, RDFS.Datatype) in self.graph or str(
                        o_range
                    ).startswith(str(XSD)):
                        is_attr = True
                        break

                name = remove_uri_prefix(str(s))
                if is_attr:
                    # Add it to attributes if not present
                    if name not in self.attributes:
                        attr_idx = len(self.attributes)
                        self.attributes[name] = Attribute(attr_idx, name)
                    self.constraints.append(
                        Constraint(
                            name=f"functional_{name}",
                            constraint_type=OWL.FunctionalProperty,
                            terms=[self.attributes[name]],
                        )
                    )
                else:
                    # Add it to relations if not present
                    if name not in self.relations:
                        rel_idx = len(self.relations)
                        self.relations[name] = Relation(rel_idx, name)
                    self.constraints.append(
                        Constraint(
                            name=f"functional_{name}",
                            constraint_type=OWL.FunctionalProperty,
                            terms=[self.relations[name]],
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
                                terms=[sub_prop],  # Add constraint for hasMother, etc.
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

        WORK IN PROGRESS AS OF 08/11/2025.
        """

        # TODO implement more complex parsing here

        # Example for owl:someValuesFrom
        # [a owl:Restriction; owl:onProperty P; owl:someValuesFrom C] rdfs:subClassOf ClassA
        # This means: (X, P, Y), (Y, type, C) -> (X, type, ClassA)
        if isinstance(s, BNode):
            try:
                # Check for: (s, rdf:type, owl:Restriction)
                prop = next(self.graph.objects(s, OWL.onProperty))
                some_class = next(self.graph.objects(s, OWL.someValuesFrom))

                rel_p = self.get_object_property(prop)  # Can only be ObjectProperty
                class_c = self.get_class(some_class)
                class_a = self.get_class(o)  # The conclusion

                if rel_p and class_c and class_a:
                    rule = ExecutableRule(
                        name=f"someValuesFrom_{rel_p.name}_{class_c.name}_{class_a.name}",
                        conclusion=GoalPattern(Var("X"), RDF.type, class_a),
                        premises=[
                            GoalPattern(Var("X"), rel_p, Var("Y")),
                            GoalPattern(Var("Y"), RDF.type, class_c),
                        ],
                    )
                    self.rules.append(rule)

            except StopIteration:
                pass

        # ... other complex axioms like allValuesFrom, hasValue, etc. can be added here
