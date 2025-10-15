import json
import re
import unicodedata
from copy import deepcopy
from importlib import resources
from pathlib import Path

from platformdirs import user_data_dir

from .abstract_config import AbstractConfig

PROJECT_FILE = "project.json"

DEFAULT_PROJECT_CONFIG = {
    "name": "",
    "graphUri": "",
    "graphNamespace": "",
    "dbUrl": "",
    "isLocal": False,  # whether the project is local or a sparql endpoint
}

PROJECT_SCHEMA = {
    "name": str,
    "graphUri": str,
    "graphNamespace": str,
    "dbUrl": str,
    "isLocal": bool,
}


def safe_filename(input_string, replacement='_', max_length=255):
    normalized = unicodedata.normalize('NFKD', input_string).encode('ascii', 'ignore').decode()
    # Replace invalid characters with the replacement
    safe = re.sub(r'[^a-zA-Z0-9._-]', replacement, normalized)
    safe = re.sub(f'{re.escape(replacement)}+', replacement, safe).strip(replacement)
    return safe[:max_length]


class Project(AbstractConfig):
    def __init__(self, project_name):
        # TODO pass app_name as parameter (not really needed, but for consistency)
        self.project_path = Path(user_data_dir("threedont")) / "projects" / f"{safe_filename(project_name)}"
        self.project_path.mkdir(parents=True, exist_ok=True)
        config = deepcopy(DEFAULT_PROJECT_CONFIG)
        config["name"] = project_name
        super().__init__(self.project_path / PROJECT_FILE, config, PROJECT_SCHEMA, auto_save=False)

    def write_config_to_file(self, file):
        json.dump(self.config, file, indent=2)

    def read_config_from_file(self, file):
        return json.load(file)

    @staticmethod
    def get_project_list():
        projects_dir = Path(user_data_dir("threedont")) / "projects"
        if not projects_dir.exists():
            return []
        # get all directories in the projects directory
        return [Project(p.name).get_name() for p in projects_dir.iterdir() if
                p.is_dir() and (p / PROJECT_FILE).exists()]

    @staticmethod
    def exists(project_name):
        project_path = Path(user_data_dir("threedont")) / "projects" / f"{safe_filename(project_name)}"
        return (project_path / PROJECT_FILE).exists()

    def get_onto_path(self):
        # ugly way for now
        assets_folder = resources.files("threedont.assets")
        namespace = self.get_graphNamespace().lower()
        if "heritage" in namespace:
            path = assets_folder / "Heritage_Ontology.rdf"
        elif "urban" in namespace:
            path = assets_folder / "Urban_Ontology.rdf"
        else:
            raise Exception("Namespace not recognized: " + namespace)  # TODO make better
        if not path.exists():
            raise Exception("Path should exists but doesn't: " + str(path))  # TODO make better also here
        return str(path)

    def get_storage_path(self):
        if not self.get_isLocal():
            raise Exception("Storage path is only available for local projects.")

        return self.project_path / "storage"
