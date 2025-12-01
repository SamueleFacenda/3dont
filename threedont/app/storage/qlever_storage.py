from .storage_factory import StorageFactory
from .abstract_storage import AbstractStorage
from .query_handlers import Query

from .pyqlever import Qlever, QleverQueryResult


class QleverQuery(QleverQueryResult, Query):
    def __init__(self, storage, query):
        Query.__init__(self, query, chunked=False)
        QLeverQueryResult.__init__(self, storage)

@StorageFactory.register(is_local=True, priority=5)
class QleverStorage(Qlever, AbstractStorage):
    def query(self, query: str, chunked: bool = True):
        return QleverQuery(self, query)


