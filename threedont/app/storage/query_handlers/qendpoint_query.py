from .query_result import Query

class QendpointQuery(Query):
    """
    A class to handle qendpoint queries.
    """
    def __init__(self, repo, query):
        self.results = {}
        self.repo = repo
        super().__init__(query, False)

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
        with self.repo.executeTupleQuery(query, 100000) as execute: # Unbounded query time
            result = execute.getResult()
            vars = result.getBindingNames()
            for var in vars:
                self.results[var] = []

            for row in result:
                for var in vars:
                    self.results[var].append(str(row.getValue(var).stringValue()))
        return self.results, self._result_len(self.results)

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