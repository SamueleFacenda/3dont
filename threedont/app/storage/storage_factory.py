from .abstract_storage import AbstractStorage
from ..state import Project

class StorageFactory:
    """
    Factory class for creating storage instances based on the provided storage type.
    _registry_local: dict[int, Type[AbstractStorage]] where int is the priority of the storage type, for local storage.
    _registry_remote: dict[int, Type[AbstractStorage]] where int is the priority of the storage type, for remote storage (sparql endpoints).
    """

    _registry_local = {}
    _registry_remote = {}

    @classmethod
    def register(cls, is_local: bool, priority: int):
        def inner_register(storage_class):
            if is_local:
                cls._registry_local[priority] = storage_class
            else:
                cls._registry_remote[priority] = storage_class
            return storage_class
        return inner_register

    @classmethod
    def create(cls, project: 'Project') -> 'AbstractStorage':
        """
        Create a storage instance based on the project configuration.

        :param project: The project configuration containing storage details.
        :return: An instance of AbstractStorage.
        """
        if project.get_isLocal():
            # Local storage
            for priority in sorted(cls._registry_local.keys()):
                storage_class = cls._registry_local[priority]
                return storage_class(project)
        else:
            # Remote storage
            for priority in sorted(cls._registry_remote.keys()):
                storage_class = cls._registry_remote[priority]
                return storage_class(project)
        raise ValueError("No suitable storage class found for the project configuration.")
