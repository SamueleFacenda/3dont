from SPARQLWrapper import TURTLE
from .turtle_result_parser import SPARQLWrapperWithTurtle

from .abstract_storage import AbstractStorage
from .storage_factory import StorageFactory
from .query_handlers import SparqlQuery

@StorageFactory.register(is_local=False, priority=10)
class SparqlWrapperStorage(AbstractStorage):
    """
    Storage class that uses SPARQLWrapper to interact with SPARQL endpoints.
    This class is designed to handle SPARQL queries and updates using the SPARQLWrapper library.
    """

    def setup_storage(self, identifier: str = None, endpoint: str = None):
        """
        Setup the SPARQLWrapper with the given endpoint and identifier.
        """
        self.sparql = SPARQLWrapperWithTurtle(endpoint)
        self.sparql.setReturnFormat(TURTLE)

    def query(self, query: str, chunked: bool = True):
        return SparqlQuery(self.sparql, query, chunked=chunked)

    def update(self, query: str):
        self.sparql.setQuery(query)
        self.sparql.query()

    def is_empty(self) -> bool:
        raise NotImplementedError("is_empty is not implemented for SparqlWrapperStorage. Use query to check for data presence.")

    def bind_to_path(self, path: str):
        raise NotImplementedError("bind_to_path is not applicable for SparqlWrapperStorage. Use the SPARQL endpoint to load data.")

    def load_file(self, path: str):
        raise NotImplementedError("load_file is not applicable for SparqlWrapperStorage. Use the SPARQL endpoint to load data.")