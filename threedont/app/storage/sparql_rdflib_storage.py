from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore
from rdflib import Graph

from .abstract_storage import AbstractStorage
from .storage_factory import StorageFactory
from .query_handlers import RdfQuery

@StorageFactory.register(is_local=False, priority=40)
class SparqlRDFLibStorage(AbstractStorage):
    def setup_storage(self, identifier: str = None, endpoint: str = None):
        self.graph = Graph(store=SPARQLUpdateStore(endpoint, endpoint, returnFormat='csv'), identifier=identifier)

    def query(self, query: str, chunked: bool = True):
        return RdfQuery(self.graph, query, chunked=chunked)

    def update(self, query: str):
        self.graph.update(query)

    def is_empty(self) -> bool:
        raise NotImplementedError("is_empty is not implemented for SparqlRDFLibStorage. Use query to check for data presence.")

    def bind_to_path(self, path: str):
        raise NotImplementedError("bind_to_path is not applicable for SparqlRDFLibStorage.")

    def load_file(self, path: str):
        raise NotImplementedError("load_file is not applicable for SparqlRDFLibStorage. Use the SPARQL endpoint to load data.")