"""
DESCRIPTION:

    Ontology Parser (OntologyParser)

    Parses an OWL/RDFS ontology file (e.g., .ttl) and translates
    OWL 2 RL axioms into executable rules and constraints for
    the backward chainer.

    It populates:
        - self.rules: List[ExecutableRule]
        - self.constraints: List[Constraint]
        - self.classes: Dict[str, Class]
        - self.relations: Dict[str, Relation]
        - self.attributes: Dict[str, Attribute]

AUTHOR

    Based on user request.
    Improved by Gemini.
"""

from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from typing import Dict, List, Set, Optional, Tuple, Union

# Custom imports from your provided files
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
    Var,
    ExecutableRule,
    Constraint,
)


class OntologyParser:
    """
    Parses an ontology file and extracts schema, rules, and constraints.
    """

    def __init__(self, ontology_file: str):
        """
        Initializes the parser, loads the graph, and parses the ontology.

        Args:
            ontology_file (str): Path to the .ttl ontology file.
        """
        self.graph = Graph()
        try:
            self.graph.parse(ontology_file, format="turtle")
        except Exception as e:
            print(f"Error parsing ontology file: {e}")
            raise

        # Storage for schema elements
        self.classes: Dict[str, Class] = {}
        self.relations: Dict[str, Relation] = {}
        self.attributes: Dict[str, Attribute] = {}

        # Index counters
        self._class_idx = 0
        self._rel_idx = 0
        self._attr_idx = 0

        # Storage for parsed rules and constraints
        self.rules: List[ExecutableRule] = []
        self.constraints: List[Constraint] = []

        # Re-usable variables for rule creation
        self.X = Var("X")
        self.Y = Var("Y")
        self.Z = Var("Z")

        # --- Main parsing pipeline ---
        print("Discovering schema (Classes, Properties)...")
        self._discover_schema()
        print("Parsing ontology axioms into rules and constraints...")
        self._setup_handlers()
        self._parse_rules_and_constraints()
        print("Ontology parsing complete.")

    def _get_clean_name(self, uri: URIRef) -> str:
        """
        Removes the URI prefix to get a clean name.
        e.g., "http://example.org/family#hasParent" -> "hasParent"
        """
        if (
            isinstance(uri, BNode)
            or isinstance(uri, Literal)
            or not isinstance(uri, URIRef)
        ):
            return str(uri)

        name_str = str(uri)

        # Try to use rdflib's built-in qname (prefixed name)
        try:
            qname = self.graph.namespace_manager.qname(uri)
            if ":" in qname and not qname.startswith("http"):
                return qname.split(":", 1)[-1]
        except:
            pass  # Fallback to manual split

        # Manual fallback
        if "#" in name_str:
            return name_str.split("#")[-1]

        return name_str.split("/")[-1]

    # --- Schema Get-or-Create Methods ---

    def _get_class(self, uri: URIRef) -> Class:
        """Gets or creates a Class object from a URI."""
        name = self._get_clean_name(uri)
        if name not in self.classes:
            self.classes[name] = Class(index=self._class_idx, name=name)
            self._class_idx += 1
        return self.classes[name]

    def _get_relation(self, uri: URIRef) -> Relation:
        """Gets or creates a Relation (ObjectProperty) object from a URI."""
        name = self._get_clean_name(uri)
        if name not in self.relations:
            self.relations[name] = Relation(index=self._rel_idx, name=name)
            self._rel_idx += 1
        return self.relations[name]

    def _get_attribute(self, uri: URIRef) -> Attribute:
        """Gets or creates an Attribute (DatatypeProperty) object from a URI."""
        name = self._get_clean_name(uri)
        if name not in self.attributes:
            self.attributes[name] = Attribute(index=self._attr_idx, name=name)
            self._attr_idx += 1
        return self.attributes[name]

    def _get_term(self, uri: URIRef) -> Union[Relation, Attribute]:
        """
        Gets or creates a property (Relation or Attribute).
        Checks the graph for its type. Defaults to Relation if unknown.
        """
        name = self._get_clean_name(uri)
        if name in self.relations:
            return self.relations[name]
        if name in self.attributes:
            return self.attributes[name]

        # If unknown, check its type in the graph
        if (uri, RDF.type, OWL.DatatypeProperty) in self.graph:
            return self._get_attribute(uri)

        # Default to Relation (ObjectProperty)
        return self._get_relation(uri)

    def _discover_schema(self) -> None:
        """
        Pre-populates the schema dicts by finding explicit declarations
        of classes and properties in the graph.
        """
        # Find Classes
        for s in self.graph.subjects(predicate=RDF.type, object=OWL.Class):
            if isinstance(s, URIRef):
                self._get_class(s)
        for s in self.graph.subjects(predicate=RDF.type, object=RDFS.Class):
            if isinstance(s, URIRef):
                self._get_class(s)

        # Find Object Properties (Relations)
        for s in self.graph.subjects(predicate=RDF.type, object=OWL.ObjectProperty):
            if isinstance(s, URIRef):
                self._get_relation(s)

        # Find Datatype Properties (Attributes)
        for s in self.graph.subjects(predicate=RDF.type, object=OWL.DatatypeProperty):
            if isinstance(s, URIRef):
                self._get_attribute(s)

    def _parse_rdf_list(self, node: BNode) -> List[URIRef]:
        """Helper to parse an RDF list (used for property chains)."""
        chain: List[URIRef] = []
        curr = node
        while curr and curr != RDF.nil:
            item = self.graph.value(subject=curr, predicate=RDF.first)
            if item and isinstance(item, URIRef):
                chain.append(item)
            curr = self.graph.value(subject=curr, predicate=RDF.rest)
        return chain

    def _setup_handlers(self) -> None:
        """Initializes the predicate-to-handler mapping."""
        self.handlers = {
            # RDFS Rules
            RDFS.subClassOf: self._handle_subClassOf,
            RDFS.subPropertyOf: self._handle_subPropertyOf,
            RDFS.domain: self._handle_domain,
            RDFS.range: self._handle_range,
            # OWL 2 RL Property Rules
            OWL.inverseOf: self._handle_inverseOf,
            OWL.propertyChainAxiom: self._handle_propertyChainAxiom,
            # OWL 2 RL Constraints / Type-based rules
            RDF.type: self._handle_rdf_type,
            OWL.disjointWith: self._handle_disjointWith,
        }

    def _parse_rules_and_constraints(self) -> None:
        """
        Main parsing loop. Iterates all triples and calls the appropriate
        handler based on the predicate.
        """
        for s, p, o in self.graph:
            # We only care about triples where the predicate is a URI
            if not isinstance(p, URIRef):
                continue

            # Find the handler for this predicate
            handler = self.handlers.get(p)
            if handler:
                try:
                    # Call the specific handler
                    handler(s, p, o)
                except Exception as e:
                    print(f"Warning: Error handling triple ({s}, {p}, {o}): {e}")
            else:
                print(
                    f"Info: No handler for predicate {p}, skipping triple ({s}, {p}, {o})"
                )

    # --- Individual Axiom Handlers ---

    def _handle_subClassOf(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            # cax-sco: (C1 rdfs:subClassOf C2) -> (?X rdf:type C1) -> (?X rdf:type C2)
            c1 = self._get_class(s)
            c2 = self._get_class(o)
            rule = ExecutableRule(
                name=f"rdfs_subClassOf_{c1.name}_{c2.name}",
                conclusion=Atom(self.X, RDF.type, c2),
                premises=[Atom(self.X, RDF.type, c1)],
            )
            self.rules.append(rule)

    def _handle_subPropertyOf(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            # cax-spo: (P1 rdfs:subPropertyOf P2) -> (?X P1 ?Y) -> (?X P2 ?Y)
            p1 = self._get_term(s)
            p2 = self._get_term(o)
            rule = ExecutableRule(
                name=f"rdfs_subPropertyOf_{p1.name}_{p2.name}",
                conclusion=Atom(self.X, p2, self.Y),
                premises=[Atom(self.X, p1, self.Y)],
            )
            self.rules.append(rule)

    def _handle_domain(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            # prp-dom: (P rdfs:domain C) -> (?X P ?Y) -> (?X rdf:type C)
            prop = self._get_term(s)
            cls = self._get_class(o)
            rule = ExecutableRule(
                name=f"rdfs_domain_{prop.name}_{cls.name}",
                conclusion=Atom(self.X, RDF.type, cls),
                premises=[Atom(self.X, prop, self.Y)],
            )
            self.rules.append(rule)

    def _handle_range(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            # prp-rng: (P rdfs:range C) -> (?X P ?Y) -> (?Y rdf:type C)
            prop = self._get_term(s)
            cls = self._get_class(o)
            rule = ExecutableRule(
                name=f"rdfs_range_{prop.name}_{cls.name}",
                conclusion=Atom(self.Y, RDF.type, cls),
                premises=[Atom(self.X, prop, self.Y)],
            )
            self.rules.append(rule)

    def _handle_inverseOf(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            # prp-inv1: (P1 owl:inverseOf P2) -> (?X P1 ?Y) -> (?Y P2 ?X)
            p1 = self._get_relation(s)
            p2 = self._get_relation(o)
            rule1 = ExecutableRule(
                name=f"owl_inverseOf_{p1.name}_{p2.name}",
                conclusion=Atom(self.Y, p2, self.X),
                premises=[Atom(self.X, p1, self.Y)],
            )
            # prp-inv2: (P1 owl:inverseOf P2) -> (?Y P2 ?X) -> (?X P1 ?Y)
            rule2 = ExecutableRule(
                name=f"owl_inverseOf_{p2.name}_{p1.name}",
                conclusion=Atom(self.X, p1, self.Y),
                premises=[Atom(self.Y, p2, self.X)],
            )
            self.rules.append(rule1)
            self.rules.append(rule2)

    def _handle_propertyChainAxiom(self, s: URIRef, p: URIRef, o: BNode) -> None:
        if isinstance(s, URIRef) and isinstance(o, BNode):
            # prp-prp-chain: (P owl:propertyChainAxiom (P1 ... Pn))
            p_chain = self._get_relation(s)
            chain_list = self._parse_rdf_list(o)

            if len(chain_list) == 2:
                # Specific handler for chain of 2
                # (?X P1 ?Y), (?Y P2 ?Z) -> (?X P_chain ?Z)
                p1 = self._get_relation(chain_list[0])
                p2 = self._get_relation(chain_list[1])
                rule = ExecutableRule(
                    name=f"owl_chain_{p1.name}_{p2.name}_{p_chain.name}",
                    conclusion=Atom(self.X, p_chain, self.Z),
                    premises=[Atom(self.X, p1, self.Y), Atom(self.Y, p2, self.Z)],
                )
                self.rules.append(rule)
            elif len(chain_list) == 1:
                # (P owl:propertyChainAxiom (P1)) -> (?X P1 ?Y) -> (?X P ?Y)
                p1 = self._get_relation(chain_list[0])
                rule = ExecutableRule(
                    name=f"rdfs_subPropertyOf_{p1.name}_{p_chain.name}",
                    conclusion=Atom(self.X, p_chain, self.Y),
                    premises=[Atom(self.X, p1, self.Y)],
                )
                self.rules.append(rule)
            # Note: Longer chains would require dynamic rule generation

    def _handle_rdf_type(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            return

        # --- Rules based on type ---
        if o == OWL.SymmetricProperty:
            # prp-sym: (P rdf:type owl:SymmetricProperty) -> (?X P ?Y) -> (?Y P ?X)
            prop = self._get_relation(s)
            rule = ExecutableRule(
                name=f"owl_symmetric_{prop.name}",
                conclusion=Atom(self.Y, prop, self.X),
                premises=[Atom(self.X, prop, self.Y)],
            )
            self.rules.append(rule)

        elif o == OWL.TransitiveProperty:
            # prp-trp: (P rdf:type owl:TransitiveProperty) -> (?X P ?Y), (?Y P ?Z) -> (?X P ?Z)
            prop = self._get_relation(s)
            rule = ExecutableRule(
                name=f"owl_transitive_{prop.name}",
                conclusion=Atom(self.X, prop, self.Z),
                premises=[Atom(self.X, prop, self.Y), Atom(self.Y, prop, self.Z)],
            )
            self.rules.append(rule)

        # --- Constraints based on type ---
        elif o == OWL.IrreflexiveProperty:
            # prp-irr: (P rdf:type owl:IrreflexiveProperty)
            # This is a constraint: NOT (?X P ?X)
            prop = self._get_term(s)
            constraint = Constraint(
                name=f"owl_irreflexive_{prop.name}",
                constraint_type=OWL.IrreflexiveProperty,
                terms=[prop, self.X],  # Represents (?X P ?X) is forbidden
            )
            self.constraints.append(constraint)

    def _handle_disjointWith(self, s: URIRef, p: URIRef, o: URIRef) -> None:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            # cls-disj: (C1 owl:disjointWith C2)
            # This is a constraint: NOT ((?X rdf:type C1) AND (?X rdf:type C2))
            c1 = self._get_class(s)
            c2 = self._get_class(o)
            constraint = Constraint(
                name=f"owl_disjoint_{c1.name}_{c2.name}",
                constraint_type=OWL.disjointWith,
                terms=[
                    c1,
                    c2,
                    self.X,
                ],  # Represents (?X type C1) and (?X type C2) is forbidden
            )
            self.constraints.append(constraint)

    def print_summary(self) -> None:
        """
        Prints a summary of the parsed schema, rules, and constraints.
        """
        print("\n--- Ontology Parser Summary ---")
        print(f"  Classes:    {len(self.classes)}")
        for name in self.classes:
            print(f"    - {name}")

        print(f"\n  Relations:  {len(self.relations)}")
        for name in self.relations:
            print(f"    - {name}")

        print(f"\n  Attributes: {len(self.attributes)}")
        for name in self.attributes:
            print(f"    - {name}")

        print("\n--- Parsed Rules ---")
        if not self.rules:
            print("  No rules were parsed.")
        for rule in self.rules:
            print(f"  {rule}")

        print("\n--- Parsed Constraints ---")
        if not self.constraints:
            print("  No constraints were parsed.")
        for constraint in self.constraints:
            print(f"  {constraint}")

        print("---------------------------------")
