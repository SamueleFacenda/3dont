from time import time

import numpy as np
import colorsys

from .queries import *
from .viewer import get_color_map
from .exceptions import WrongResultFormatException
from .storage import StorageFactory
from .state import Config

__all__ = ['SparqlBackend']

class SparqlBackend:
    def __init__(self, project, config):
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
        self.color_map = config.get_visualizer_scalarColorScheme()
        highlight = config.get_visualizer_highlightColor()
        # convert from FF0000 to [0.1, 0.0, 0.0]
        self.highlight_color = np.array([int(highlight[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


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
            colors[i] = self.highlight_color

        return colors

    @staticmethod
    def get_n_classes_colors(n: int):
        gr = 0.61803398875
        h = 0.0
        colors = []
        for _ in range(n):
            h = (h + gr) % 1.0
            colors.append(colorsys.hsv_to_rgb(h, 0.8, 0.8))
        return colors

    def convert_scalar_float_result(self, s, x):
        # convert to float
        results_x = np.fromiter(x, dtype=np.float32)
        minimum = results_x.min()
        maximum = results_x.max()
        print("Scalar query min: ", minimum, " max: ", maximum)
        default = minimum - (maximum - minimum) / 10
        scalars = np.full(len(self.colors), default, dtype=np.float32)
        for subject, scalar in zip(s, results_x):
            i = self.iri_to_id[subject]
            scalars[i] = scalar
        colors = get_color_map(self.color_map)
        return scalars, scalars, colors

    def convert_scalar_class_result(self, s, x):
        unique_classes = list(set(x))
        colors_list = self.get_n_classes_colors(len(unique_classes))
        class_to_color = {a:b for a, b in zip(unique_classes, colors_list)}

        if len(s) != len(set(s)):
            print("Warning: duplicate subjects in class query result, the result may be incorrect.")

        scalars = np.copy(self.colors)
        for subject, cls in zip(s, x):
            try:
                i = self.iri_to_id[subject]
            except KeyError:
                continue  # not all the results of a select are points
            scalars[i] = class_to_color[cls]

        return scalars, unique_classes, colors_list

    @staticmethod
    def is_iri(value):
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if isinstance(value, np.bytes_):
            value = value.astype(str)
        return hasattr(value, 'startswith') and (value.startswith('http') or value.startswith('<http'))

    def execute_scalar_query(self, query):
        results = self.storage.query(query)
        if not results.has_var('x') or not results.has_var('s'):
            raise WrongResultFormatException(['s', 'x'], results.vars())

        if self.is_iri(results['x'][0]):
            return self.convert_scalar_class_result(results['s'], results['x'])

        return self.convert_scalar_float_result(results['s'], results['x'])

    @staticmethod
    def uniform_iri(iri):
        if iri.startswith('http'):
            iri = '<' + iri + '>'
        return iri

    def get_point_iri(self, point_id):
        out = self.id_to_iri[point_id]
        # unify the type to str
        if isinstance(out, np.bytes_):
            out = out.astype(str)
        if isinstance(out, bytes):
            out = out.decode('utf-8')

        return out

    def get_node_details(self, iri):
        iri = self.uniform_iri(iri)
        query = GET_NODE_DETAILS.format(graph=self.graph_uri, point=iri, namespace=self.onto_namespace)
        results = self.storage.query(query, chunked=False)
        out = list(map(tuple, results.tuple_iterator(['p', 'o'])))
        return out

    @staticmethod
    def chain_predicate(predicateList):
        '''
        From ['a', 'b', 'c'] to
            a ?a1 .
        ?a1 b ?a2 .
        ?a2 c
        '''
        predicate = ''
        prev = ''
        for i, p in enumerate(predicateList[:-1]):
            current = '?a' + str(i)
            predicate += f'{prev} {p} {current} .\n'
            prev = current

        predicate += f'{prev} {predicateList[-1]} '
        return predicate

    def execute_predicate_query(self, predicateList):
        predicates = [self.uniform_iri(p) for p in predicateList]
        predicate = self.chain_predicate(predicates)

        query = PREDICATE_QUERY.format(graph=self.graph_uri, predicate=predicate, namespace=self.onto_namespace)
        return self.execute_scalar_query(query)

    def annotate_node(self, subject, predicate, object):
        subject, predicate, object = self.uniform_iri(subject), self.uniform_iri(predicate), self.uniform_iri(object)
        query = ANNOTATE_NODE.format(graph=self.graph_uri, subject=subject, predicate=predicate, object=object,
                                     namespace=self.onto_namespace)
        self.storage.update(query)

    def select_all_subjects(self, predicateList, object):
        predicates = [self.uniform_iri(p) for p in predicateList]
        predicate = self.chain_predicate(predicates)

        object = self.uniform_iri(object)
        query = SELECT_ALL_WITH_PREDICATE.format(graph=self.graph_uri, predicate=predicate, object=object,
                                                 namespace=self.onto_namespace)
        iris = self.storage.query(query)['p']
        colors = np.copy(self.colors)
        for p in iris:
            try:
                i = self.iri_to_id[p]
            except KeyError:
                continue  # not all the results of a select are points
            colors[i] = self.highlight_color
        return colors

    def raw_query(self, query):
        results = self.storage.query(query)
        header = results.vars()
        content = results.tuple_iterator(header)
        return header, content

    def autodetect_query_nl(self, query):
        # TODO refactor
        result = self.storage.query(query)
        columns = result.vars()
        if 'x1' in columns and 'y1' in columns and 'z1' in columns and len(columns) == 3:
            # select query
            print("Detected select query, columns: ", columns)
            colors = np.copy(self.colors)
            coords = np.array((result['x1'], result['y1'], result['z1'])).T.astype(np.float32)
            for coord in coords:
                try:
                    i = self.coords_to_id[tuple(coord)]
                except KeyError:
                    print("Coordinate not found in point cloud (very strange): ", coord)
                    continue  # not all the results of a select are points
                colors[i] = self.highlight_color
            return colors, "select"

        if 'x1' in columns and 'y1' in columns and 'z1' in columns:
            print("Detected scalar query, columns: ", columns)
            scalar_col = [col for col in columns if col not in ('x1', 'y1', 'z1')][0]
            coords = np.array((result['x1'], result['y1'], result['z1'])).T.astype(np.float32)
            ids = [self.coords_to_id[tuple(coord)] for coord in coords]
            if self.is_iri(result[scalar_col][0]):
                return self.convert_scalar_class_result(ids, result[scalar_col]), "scalar"

            return self.convert_scalar_float_result(ids, result[scalar_col]), "scalar"

        return result, "tabular"
