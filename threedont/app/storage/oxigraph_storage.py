from pyoxigraph import Store, NamedNode

from .storage_factory import StorageFactory
from .abstract_storage import AbstractStorage
from .query_handlers import OxigraphQuery

@StorageFactory.register(is_local=True, priority=20)
class OxigraphStorage(AbstractStorage):
    """
    Storage class that uses Oxigraph to interact with RDF data.
    This class is designed to handle RDF queries and updates using the Oxigraph library.
    """

    def setup_storage(self, identifier: str = None, endpoint: str = None):
        self.graph = None
        self.identifier = identifier

    def query(self, query: str, chunked: bool = True):
        return OxigraphQuery(self.graph, query)

    def is_empty(self) -> bool:
        return len(self.graph) == 0

    def bind_to_path(self, path: str):
        self.graph = Store(path)

    def load_file(self, path: str):
        self.graph.add_graph(NamedNode(self.identifier))
        self.graph.bulk_load(path=path, to_graph=NamedNode(self.identifier))
        self.graph.flush()

    def update(self, query: str):
        self.graph.update(query)
        self.graph.flush()
