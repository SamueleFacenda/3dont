from rdflib import Graph

from .abstract_storage import AbstractStorage
from .storage_factory import StorageFactory
from .query_result import Query

@StorageFactory.register(is_local=True, priority=10)
class OxigraphRDFLibStorage(AbstractStorage):

    def setup_storage(self, identifier: str = None, endpoint: str = None):
        self.graph = Graph(store="Oxigraph", identifier=identifier)

    def query(self, query: str, chunked: bool = True):
        return Query(self.graph, query, chunked=False)

    def is_empty(self) -> bool:
        return len(self.graph) == 0

    def bind_to_path(self, path: str):
        self.graph.open(path)

    def load_file(self, path: str):
        if path.endswith('.ttl'):
            format = 'ox-ttl'
        elif path.endswith('.rdf'):
            format = 'ox-xml'
        else:
            raise ValueError("Unsupported file format for ontology: " + path)

        self.graph.parse(path, format=format, transactional=False)

    def update(self, query: str):
        self.graph.update(query)