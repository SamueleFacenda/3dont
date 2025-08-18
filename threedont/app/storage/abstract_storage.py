from abc import ABC, abstractmethod

class AbstractStorage(ABC):
    """
    Abstract base class for sparql storage implementations.
    """

    def __init__(self, project):
        if project.get_isLocal():
            self.setup_storage(identifier=project.get_graphUri())
            self.bind_to_path(project.get_storage_path())
            if self.is_empty():
                print("The original ontology file will be imported, this may take a while...")
                self.load_file(project.get_originalPath())
        else:
            # TODO generalize outside of virtuoso
            endpoint = project.get_dbUrl() + "/sparql"
            self.setup_storage(endpoint=endpoint)

    @abstractmethod
    def setup_storage(self, identifier: str = None, endpoint: str = None):
        """
        Initialize the storage. This method should be called after the storage instance is created.
        It can be used to set up any necessary configurations or connections.
        """
        pass

    @abstractmethod
    def query(self, query: str, chunked: bool = True):
        """
        Execute a SPARQL query against the storage.

        :param query: The SPARQL query to execute.
        :param chunked: Whether to perform the query in chunks.
        :return: The results of the query.
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """
        Check if the storage is empty.

        :return: True if the storage is empty, False otherwise.
        """
        pass

    @abstractmethod
    def bind_to_path(self, path: str):
        """
        Load data from a file into the storage.

        :param path: The path to the file to load.
        """
        pass

    @abstractmethod
    def load_file(self, path: str):
        """
        Load a file into the storage.

        :param path: The path to the file to load.
        """
        pass

    @abstractmethod
    def update(self, query: str):
        """
        Execute an update query against the storage.

        :param query: The SPARQL update query to execute.
        """
        pass