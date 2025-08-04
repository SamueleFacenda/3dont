from rdflib import URIRef

from .exceptions import EmptyResultSetException

TEST_FAST = False  # remove true before commit
CHUNK_SIZE = 1000000 if not TEST_FAST else 1000

class Query:
    """
    A class to handle SPARQL query results in chunks.
    """
    def __init__(self, graph, query, chunked=True, cast_to_strings=True):
        offset = 0
        self.chunks = []
        # TODO multithreaded query execution
        if chunked:
            while True:
                chunked_query = query + " OFFSET " + str(offset) + " LIMIT " + str(CHUNK_SIZE)
                results = graph.query(chunked_query)
                self.chunks += [results]

                if len(results) < CHUNK_SIZE:
                    break
                if TEST_FAST:
                    break
                offset += CHUNK_SIZE
        else:
            results = graph.query(query)
            self.chunks = [results]

        if not self.chunks or len(self.chunks[0]) == 0:
            raise EmptyResultSetException(query)

        if cast_to_strings:
            # self.cast = str
            self.cast = lambda x: str(x) if not isinstance(x, URIRef) else f'<{x}>'
        else:
            self.cast = lambda x: x

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