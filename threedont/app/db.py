from time import time

import numpy as np

from .queries import *
from .exceptions import WrongResultFormatException
from .storage import StorageFactory

HIGHLIGHT_COLOR = [1.0, 0.0, 0.0]  # TODO make this a parameter

__all__ = ['SparqlBackend']

class SparqlBackend:
    def __init__(self, project):
        self.graph_uri = project.get_graphUri()
        onto_namespace = project.get_ontologyNamespace()
        if onto_namespace.endswith('#'):
            self.onto_namespace = onto_namespace
        else:
            self.onto_namespace = onto_namespace + "#"

        self.storage = StorageFactory.create(project)
        self.iri_to_id = {}
        self.coords_to_id = {}
        self.id_to_iri = []
        self.colors = None


    def get_all(self):
        query = SELECT_ALL_QUERY.format(graph=self.graph_uri, namespace=self.onto_namespace)
        start = time()
        results = self.storage.query(query)
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
        results = self.storage.query(query)

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
        results = self.storage.query(query)
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
        query = GET_NODE_DETAILS.format(graph=self.graph_uri, point=iri, namespace=self.onto_namespace)
        results = self.storage.query(query, chunked=False)
        out = list(results.tuple_iterator(['p', 'o']))
        return out

    def execute_predicate_query(self, predicate):
        query = PREDICATE_QUERY.format(graph=self.graph_uri, predicate=predicate, namespace=self.onto_namespace)
        return self.execute_scalar_query(query)

    def annotate_node(self, subject, predicate, object):
        query = ANNOTATE_NODE.format(graph=self.graph_uri, subject=subject, predicate=predicate, object=object,
                                     namespace=self.onto_namespace)
        self.storage.update(query)

    def select_all_subjects(self, predicate, object):
        query = SELECT_ALL_WITH_PREDICATE.format(graph=self.graph_uri, predicate=predicate, object=object,
                                                 namespace=self.onto_namespace)
        iris = self.storage.query(query)['p']
        colors = np.copy(self.colors)
        for p in iris:
            try:
                i = self.iri_to_id[p]
            except KeyError:
                continue  # not all the results of a select are points
            colors[i] = HIGHLIGHT_COLOR
        return colors

    def raw_query(self, query):
        results = self.storage.query(query)
        header = results.vars()
        content = results.tuple_iterator(header)
        return header, content

    def autodetect_query_nl(self, query):
        # TODO refactor
        result = self.storage.query(query)
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
