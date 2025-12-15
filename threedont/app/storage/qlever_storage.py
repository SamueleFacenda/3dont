from .storage_factory import StorageFactory
from .abstract_storage import AbstractStorage
from .query_handlers import Query

from .pyqlever import Qlever, QleverQueryResult

import psutil


class QleverQuery(QleverQueryResult, Query):
    def __init__(self, storage, query):
        QleverQueryResult.__init__(self, storage)
        Query.__init__(self, query, chunked=False)

@StorageFactory.register(is_local=True, priority=5)
class QleverStorage(Qlever, AbstractStorage):
    def __init__(self, project):
        # Prefix is used to compress IRIs, it's the most recurring IRIs prefix
        half_gb = psutil.virtual_memory().total // (1024 ** 3) // 2
        graph_namespace = project.get_graphNamespace()
        if not graph_namespace.endswith(('#', '/')):
            graph_namespace += '#'

        Qlever.__init__(self, prefix=graph_namespace, max_memory_gb=half_gb)
        AbstractStorage.__init__(self, project)

    def query(self, query: str, chunked: bool = True):
        return QleverQuery(self, query)


