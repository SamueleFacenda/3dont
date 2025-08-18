from .query_result import Query

class SparqlQuery(Query):
    """
    A class to handle SPARQL query results in chunks.
    """
    def __init__(self, sparql, query, chunked=True):
        self.results = {}
        self.sparql = sparql
        super().__init__(query, chunked)

    def _append_chunk(self, chunk):
        for key in chunk.keys():
            if key not in self.results:
                self.results[key] = []
            self.results[key].extend(chunk[key])

    @staticmethod
    def _result_len(result):
        if not result:
            return 0
        any_key = next(iter(result.keys()))
        return len(result[any_key])

    def _perform_query(self, query):
        self.sparql.setQuery(query)
        result = self.sparql.queryAndConvert()
        return result, self._result_len(result)

    def __len__(self):
        return self._result_len(self.results)

    def __iter__(self):
        for i in range(len(self)):
            yield {key: self.results[key][i] for key in self.results.keys()}

    def __getitem__(self, index):
        return self.results[index]

    def tuple_iterator(self, keys):
        return (tuple(self.results[key][i] for key in keys) for i in range(len(self)))

    def vars(self):
        return list(self.results.keys())

    def has_var(self, var):
        return var in self.vars()