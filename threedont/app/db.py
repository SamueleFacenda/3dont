from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE
from urllib.parse import urlparse
import numpy as np
import re
from time import time

from .queries import *

VARIABLES_REGEX = re.compile(r"res:binding\s*\[\s*res:variable\s*\"([a-z]+)\"\s*;\s*res:value\s*(\S+)\s*\]")

PREFIXES_REGEX = re.compile(r"^@prefix\s+([a-zA-Z0-9_]+):\s*<([^>]+)>\s*\.\s*", re.MULTILINE)

def substitute_prefix(prefixes, var):
    if not ':' in var:
        return var

    if var.startswith("<") and var.endswith(">"):
        return var[1:-1]

    prefix, suffix = var.split(":")
    try:
        return prefixes[prefix] + suffix
    except KeyError:
        # it's not possible to distinguish between iri with prefix and simple string with a colon
        return var

def parse_turtle_select(turtle):
    results = {}
    parsed = VARIABLES_REGEX.findall(turtle)
    prefixes = PREFIXES_REGEX.findall(turtle)
    prefixes = {prefix: iri for prefix, iri in prefixes}
    for var, value in parsed:
        var = substitute_prefix(prefixes, var)
        value = substitute_prefix(prefixes, value)
        if var not in results:
            results[var] = []
        results[var].append(value)

    return results

class SparqlEndpoint:
    def __init__(self, url):
        self.graph = url
        parsed = urlparse(url)
        # TODO generalize outside of virtuoso
        self.endpoint= parsed.scheme + "://" + parsed.netloc + "/sparql"
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(TURTLE)
        self.iri_to_id = {}
        self.id_to_iri = []
        self.colors = None

    def get_all(self):
        query = SELECT_ALL_QUERY.format(graph=self.graph)
        self.sparql.setQuery(query)
        start = time()
        results = self.sparql.queryAndConvert().decode()
        print("Time to query: ", time() - start)
        start = time()
        results = parse_turtle_select(results)
        print("Time to parse query result: ", time() - start)
        start = time()

        coords = np.array((results['x'], results['y'], results['z'])).T.astype(np.float32)
        colors = np.array((results['r'], results['g'], results['b'])).T.astype(np.float32)
        self.iri_to_id = {p: i for i, p in enumerate(results['p'])}
        self.id_to_iri = results['p']

        colors = colors / (1<<16)
        self.colors = colors
        print("Time to process query result: ", time() - start)
        return coords, colors

    # returns the colors with highlighted points
    def execute_select_query(self, where_clause):
        query = FILTER_QUERY.format(graph=self.graph, filter=where_clause)
        self.sparql.setQuery(query)
        try:
            results = self.sparql.queryAndConvert().decode()
        except Exception as e:
            print("Error executing query: ", e)
            return self.colors
        results = parse_turtle_select(results)
        colors = np.copy(self.colors)
        for p in results['p']:
            try:
                i = self.iri_to_id[p]
            except KeyError:
                print("Point not found: ", p)
                # This happens every time, it's a mistery for me why, probably virtuoso is misconfigured or something
                continue
            colors[i] = [1.0, 0.0, 0.0] # TODO make this a parameter

        return colors


    def get_point_iri(self, point_id):
        return self.id_to_iri[point_id]

    def get_node_details(self, iri):
        query = GET_NODE_DETAILS.format(graph=self.graph, point=iri)
        self.sparql.setQuery(query)
        results = self.sparql.queryAndConvert().decode()
        results = parse_turtle_select(results)

        if 'p' not in results or 'o' not in results:
            # assume empty result
            return []

        out = list(zip(results['p'], results['o']))

        return out


if __name__ == "__main__":
    sparql = SparqlEndpoint("http://localhost:8890/Nettuno")
    sparql.sparql.setReturnFormat(TURTLE)
    sparql.get_all()
    details = sparql.get_point_details(0)
    print(details)
    # coords, colors = sparql.get_all()
    # print(len(coords))
    # print(coords)