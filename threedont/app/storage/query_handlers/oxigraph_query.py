from pyoxigraph import NamedNode

from .query_result import Query

class OxigraphQuery(Query):
    """
    A class to handle SPARQL query results in chunks.
    """
    def __init__(self, store, query):
        self.result = {}
        self.store = store
        self.variables = None
        super().__init__(query, False)

    def _append_chunk(self, chunk):
        #TODO optimize (it takes too much time to append a chunk)
        from time import time
        start = time()
        for var in self.variables:
            self.result[str(var)] = [None] * len(chunk)
        for var in self.variables:
            if type(chunk[0][var]) is NamedNode:
                cast = str
            else:
                cast = lambda x: x.value
            for i, row in enumerate(chunk):
                self.result[str(var)][i] = cast(row[var])

        # strip the '?' prefix from variable names
        new_out = {}
        for key, value in self.result.items():
            new_out[key[1:]] = value

        self.result = new_out
        print("Time to append chunk: ", time() - start)

    def _perform_query(self, query):
        out = self.store.query(query)
        self.store.flush()
        self.variables = out.variables
        out = list(out)
        return out, len(out)

    def __len__(self):
        if not self.result:
            return 0
        any_key = next(iter(self.result))
        return len(self.result[any_key])

    def __iter__(self):
        for i in range(len(self)):
            yield tuple(self.result[var][i] for var in self.result)

    def __getitem__(self, index):
        return self.result[index]

    def tuple_iterator(self, keys):
        return zip(*(self.result[key] for key in keys))

    def vars(self):
        return list(self.result.keys())

    def has_var(self, var):
        return var in self.vars()