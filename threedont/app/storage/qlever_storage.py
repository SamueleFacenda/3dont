from .storage_factory import StorageFactory
from .abstract_storage import AbstractStorage
from .query_handlers import Query

from .pyqlever import QLever, QleverQueryResult

class QleverQuery(QleverQueryResult, Query):
    def __init__(self, storage, query):
        QLeverQueryResult.__init__(self, storage)
        Query.__init__(self, self.get_query(), chunked=False)

@StorageFactory.register(is_local=True, priority=5)
class QleverStorage(Qlever, AbstractStorage):
    def query(self, query: str, chunked: bool = True):
        return QleverQuery(self, query)


