from rdflib import URIRef

from .query_result import Query


class RdfQuery(Query):
    """
    A class to handle SPARQL query results in chunks.
    """
    def __init__(self, graph, query, chunked=True, cast_to_strings=True):
        self.graph = graph
        self.chunks = []
        super().__init__(query, chunked)

        if cast_to_strings:
            # self.cast = str
            self.cast = lambda x: str(x) if not isinstance(x, URIRef) else f'<{x}>'
        else:
            self.cast = lambda x: x

    def _append_chunk(self, chunk):
        self.chunks.append(chunk)

    def _perform_query(self, query):
        results = self.graph.query(query)
        result_len = len(results)
        return results, result_len

    def __len__(self):
        return sum(len(chunk) for chunk in self.chunks)

    def __iter__(self):
        for chunk in self.chunks:
            for row in chunk:
                yield row

    def __getitem__(self, index):
        """
            Slices not supported for now, indexing not supported for now.
        """
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not supported for Query results.")
        elif isinstance(index, int):
            raise NotImplementedError("Indexing is not supported for Query results.")
        elif isinstance(index, str):
            return (self.cast(row[index]) for chunk in self.chunks for row in chunk)
        else:
            raise TypeError(f"Unsupported index type: {type(index)}. Use string keys to access columns.")

    def tuple_iterator(self, keys):
        return (tuple(self.cast(row[key]) for key in keys) for chunk in self.chunks for row in chunk)

    def vars(self):
        """
        Returns the variable names in the query result.
        """
        if not self.chunks:
            return []
        return [self.cast(var) for var in self.chunks[0].vars]

    def has_var(self, var):
        """
        Checks if the query result contains a specific variable.
        """
        return var in self.vars()
