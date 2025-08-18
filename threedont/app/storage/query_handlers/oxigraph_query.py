from .query_result import Query

class OxigraphQuery(Query):
    """
    A class to handle SPARQL query results in chunks.
    """
    def __init__(self, store, query):
        self.result = {}
        self.store = store
        self.len = 0
        super().__init__(query, False)

    def _append_chunk(self, chunk):
        """
        Important: the variable names have a '?' prefix, which is stripped in the result.
        """
        for var in chunk.variables:
            self.result[str(var)] = [None] * self.len
        for i, row in enumerate(chunk):
            for var in chunk.variables:
                self.result[str(var)][i] = str(row[var])

        # strip the '?' prefix from variable names
        new_out = {}
        for key, value in self.result.items():
            new_out[key[1:]] = value

        self.result = new_out

    def _perform_query(self, query):
        out = self.store.query(query)
        self.len = sum(1 for _ in out)
        return out, self.len

    def __len__(self):
        return self.len

    def __iter__(self):
        for i in range(self.len):
            yield tuple(self.result[var][i] for var in self.result)

    def __getitem__(self, index):
        return self.result[index]

    def tuple_iterator(self, keys):
        return zip(*(self.result[key] for key in keys))

    def vars(self):
        return list(self.result.keys())

    def has_var(self, var):
        return var in self.vars()