from abc import ABC, abstractmethod

from threedont.app.exceptions import EmptyResultSetException

TEST_FAST = False  # remove true before commit
CHUNK_SIZE = 1000000 if not TEST_FAST else 1000


class Query(ABC):
    """
    A class to handle query results in chunks.
    """
    def __init__(self, query, chunked=True):
        offset = 0
        # TODO multithreaded query execution
        if chunked:
            while True:
                chunked_query = query + " OFFSET " + str(offset) + " LIMIT " + str(CHUNK_SIZE)
                results, result_len = self._perform_query(chunked_query)
                self._append_chunk(results)

                if result_len < CHUNK_SIZE:
                    break
                if TEST_FAST:
                    break
                offset += CHUNK_SIZE
        else:
            results, _ = self._perform_query(query)
            self._append_chunk(results)

        if len(self) == 0:
            raise EmptyResultSetException(query)


    @abstractmethod
    def _append_chunk(self, chunk):
        """
        Append a chunk of results to the query result.
        """
        pass

    @abstractmethod
    def _perform_query(self, query):
        """
        Perform the SPARQL query and return the results.
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Returns the total number of rows in the query result.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Iterate over the query result rows.
        """
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        Access the specific column / variable list
        """
        pass

    @abstractmethod
    def tuple_iterator(self, keys):
        """
        Returns an iterator over tuples of values for the specified keys.
        """
        pass

    @abstractmethod
    def vars(self):
        """
        Returns the variable names in the query result.
        """
        pass

    @abstractmethod
    def has_var(self, var):
        """
        Checks if the query result contains a specific variable.
        """
        pass
