from time import time

import numpy as np
from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore

from .queries import *
from .exceptions import WrongResultFormatException
from .query_result import Query

HIGHLIGHT_COLOR = [1.0, 0.0, 0.0]  # TODO make this a parameter

__all__ = ['SparqlBackend']

class SparqlBackend:
    def __init__(self, project):
        self.graph_uri = project.get_graphUri()
        namespace = project.get_graphNamespace()
        if namespace.endswith('#'):
            self.namespace = namespace
        else:
            self.namespace = namespace + "#"
        if project.get_isLocal():
            path = project.get_storage_path()
            self.graph = Graph(store="Oxigraph", identifier=self.graph_uri)
            self.graph.open(path, create=True)
            if not self.graph:
                source_name = project.get_onto_path() # TODO adjust
                if source_name.endswith('.ttl'):
                    format = 'ox-ttl'
                elif source_name.endswith('.rdf'):
                    format = 'ox-xml'
                else:
                    raise ValueError("Unsupported file format for ontology: " + source_name)

                self.graph.parse(source_name, format=format)
        else:
            # TODO generalize outside of virtuoso
            self.endpoint = project.get_dbUrl() + "/sparql"
            store = SPARQLUpdateStore(self.endpoint, self.endpoint, returnFormat='csv')
            self.graph = Graph(store=store)
        self.iri_to_id = {}
        self.coords_to_id = {}
        self.id_to_iri = []
        self.colors = None


    def get_all(self):
        query = SELECT_ALL_QUERY.format(graph=self.graph_uri, namespace=self.namespace)
        start = time()
        results = Query(self.graph, query)
        print("Time to query: ", time() - start)
        start = time()

        coords = np.fromiter(results.tuple_iterator(['x', 'y', 'z']), dtype=np.dtype((np.float32, 3)), count=len(results))
        colors = np.fromiter(results.tuple_iterator(['r', 'g', 'b']), dtype=np.dtype((np.float32, 3)), count=len(results))
        self.iri_to_id = {p: i for i, p in enumerate(results['p'])}
        self.coords_to_id = {tuple(c): i for i, c in enumerate(coords)}
        self.id_to_iri = list(results['p'])

        if colors.max() > 255:
            colors = colors / (1 << 16)  # 16 bit color
        else:
            colors = colors / (1 << 8)  # 8 bit color
        self.colors = colors
        print("Time to process query result: ", time() - start)
        return coords, colors

    # returns the colors with highlighted points
    def execute_select_query(self, query):
        results = Query(self.graph, query)

        colors = np.copy(self.colors)
        if not results.has_var('p'):
            raise WrongResultFormatException(['p'], results.vars())

        for p in results['p']:
            try:
                i = self.iri_to_id[p]
            except KeyError:
                continue  # not all the results of a select are points
            colors[i] = HIGHLIGHT_COLOR

        return colors

    def execute_scalar_query(self, query):
        results = Query(self.graph, query)
        if not results.has_var('x') or not results.has_var('s'):
            raise WrongResultFormatException(['s', 'x'], results.vars())

        # convert to float
        results_x = np.fromiter(results['x'], dtype=np.float32)
        minimum = results_x.min()
        maximum = results_x.max()
        print("Scalar query min: ", minimum, " max: ", maximum)
        default = minimum - (maximum - minimum) / 10
        scalars = np.full(len(self.colors), default, dtype=np.float32)
        for subject, scalar in zip(results['s'], results_x):
            i = self.iri_to_id[subject]
            scalars[i] = scalar
        return scalars

    def get_point_iri(self, point_id):
        return self.id_to_iri[point_id]

    def get_node_details(self, iri):
        query = GET_NODE_DETAILS.format(graph=self.graph_uri, point=iri, namespace=self.namespace)
        results = Query(self.graph, query, chunked=False)
        out = list(results.tuple_iterator(['p', 'o']))
        return out

    def execute_predicate_query(self, predicate):
        query = PREDICATE_QUERY.format(graph=self.graph_uri, predicate=predicate, namespace=self.namespace)
        return self.execute_scalar_query(query)

    def annotate_node(self, subject, predicate, object):
        query = ANNOTATE_NODE.format(graph=self.graph_uri, subject=subject, predicate=predicate, object=object,
                                     namespace=self.namespace)
        # TODO use update endpoint instead of query
        self.graph.update(query)

    def select_all_subjects(self, predicate, object):
        query = SELECT_ALL_WITH_PREDICATE.format(graph=self.graph_uri, predicate=predicate, object=object,
                                                 namespace=self.namespace)
        iris = Query(self.graph, query)['p']
        colors = np.copy(self.colors)
        for p in iris:
            try:
                i = self.iri_to_id[p]
            except KeyError:
                continue  # not all the results of a select are points
            colors[i] = HIGHLIGHT_COLOR
        return colors

    def raw_query(self, query):
        results = Query(self.graph, query)
        header = results.vars()
        content = results.tuple_iterator(header)
        return header, content

    def autodetect_query_nl(self, query):
        # TODO refactor
        result = Query(self.graph, query)
        columns = list(result.keys())
        if 'x1' in columns and 'y1' in columns and 'z1' in columns:
            # select query
            print("Detected select query, columns: ", columns)
            colors = np.copy(self.colors)
            coords = np.array((result['x1'], result['y1'], result['z1'])).T.astype(np.float32)
            for coord in coords:
                try:
                    i = self.coords_to_id[tuple(coord)]
                except KeyError:
                    continue  # not all the results of a select are points
                colors[i] = HIGHLIGHT_COLOR
            return colors, "select"

        if 'x1' in columns and 'y1' in columns and 'z1' in columns:
            print("Detected scalar query, columns: ", columns)
            minimum = float(min(result['x1']))
            maximum = float(max(result['x1']))
            default = minimum - (maximum - minimum) / 10
            scalars = np.full(len(self.colors), default, dtype=np.float32)
            coords = np.array((result['x1'], result['y1'], result['z1'])).T.astype(np.float32)
            for coord, scalar in zip(coords, result['x1']):
                i = self.coords_to_id[tuple(coord)]
                scalars[i] = scalar
            return scalars, "scalar"

        return result, "tabular"
