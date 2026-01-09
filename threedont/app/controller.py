import functools
import logging
import sys
from math import pi
from queue import Queue
import concurrent.futures
import multiprocessing
from urllib.error import URLError

import owlready2 as owl2
from .db import SparqlBackend
from .exceptions import WrongResultFormatException, EmptyResultSetException
from .state import Project
from .viewer import Viewer, get_color_map
from ..gui import GuiWrapper
from ..nl_2_sparql import nl_2_sparql, init_client
from ..sensor_manager import SensorArgs, sensor_management_functions as smf, aws_iot_interface as aws

__all__ = ["Controller"]

"""
    The commands_pipe will transport function calls from the GUI to the Controller.
    A functions here is a tuple of the form (function_name, args).
    ActionController is just a middleman to help with the transport between the processes, a facade.
"""

NUMBER_OF_LABELS_IN_LEGEND = 5


class ActionController:
    def __init__(self, commands_queue, start_func):
        self.commands_queue = commands_queue
        self._start = start_func

    def start(self):
        self._start()

    def __getattr__(self, item):
        # check if controller has the function
        if not hasattr(Controller, item) or not callable(getattr(Controller, item)):
            raise AttributeError(f"Controller has no method {item}")

        f = lambda *args: self.commands_queue.put((item, args))
        return f


def report_errors_to_gui(func):
    """
    Decorator to report errors to the GUI.
    """

    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except URLError as e:
            self.gui.set_statusbar_content(f"Connection error: {e}", 5)
            raise e
        except ValueError as e:
            if "Response:\n" in str(e): # Sparql query error
                message = str(e).split("Response:\n")[0]
                message = f"Bad query: {message}"
            else: # Probably qlever error
                message = str(e)
            self.gui.set_query_error(message)
            raise e
        except WrongResultFormatException as e:
            self.gui.set_query_error(f"Wrong result format: {e}")
            raise e
        except EmptyResultSetException as e:
            self.gui.set_query_error(f"Empty result set: {e}")
            raise e

    return wrapper


class Controller:
    def __init__(self, config, app_state):
        self.config = config
        self.app_state = app_state
        self.commands_queue = Queue()
        action_controller = ActionController(self.commands_queue, self.run_event_loop)
        self.gui = GuiWrapper(action_controller, sys.argv) # argv is used by qt for qt cli args
        self.viewer_client = Viewer(self.gui.send_viewer_command)
        self.sparql_client = None
        self.sensorArgs = SensorArgs()
        self.project = None

    def stop(self):
        print("Stopping controller...")
        self.commands_queue.put(None)

    def run(self):
        # this will create a thread that runs `run_event_loop`
        self.gui.run()


    @staticmethod
    def log_future_exception(future, function_name):
        try:
            future.result()
        except Exception:
            logging.exception("Error in controller running function %s", function_name)


    def run_event_loop(self):
        print("Running controller")
        self.update_project_list()
        command = None
        if self.config.get_general_loadLastProject():
            last_project = self.app_state.get_projectName()
            if last_project and Project.exists(last_project):
                command = ("open_project", (last_project,))

        if command is None:
            command = self.commands_queue.get()
        print("Controller: starting event loop")
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            while command is not None:
                function_name, args = command
                future = executor.submit(getattr(self, function_name), *args)
                future.add_done_callback(functools.partial(self.log_future_exception, function_name=function_name))
                command = self.commands_queue.get()

    @report_errors_to_gui
    def select_query(self, query):
        print("Controller: ", query)
        if self.sparql_client is None:
            print("No connection to server")
            return

        selected_colors = self.sparql_client.execute_select_query(query)
        self.viewer_client.attributes(self.sparql_client.colors, selected_colors)
        self.viewer_client.set(curr_attribute_id=1)

    @report_errors_to_gui
    def scalar_query(self, query):
        print("Controller: ", query)
        if self.sparql_client is None:
            print("No connection to server")
            return

        scalars = self.sparql_client.execute_scalar_query(query)
        self.viewer_client.attributes(self.sparql_client.colors, scalars)
        self.viewer_client.set(curr_attribute_id=1)
        self._send_legend(scalars)

    def scalar_with_predicate(self, predicate):
        print("Controller: ", predicate)
        if self.sparql_client is None:
            print("No connection to server")
            return

        scalars = self.sparql_client.execute_predicate_query(predicate)
        self.viewer_client.attributes(self.sparql_client.colors, scalars)
        self.viewer_client.set(curr_attribute_id=1)
        self._send_legend(scalars)

    @report_errors_to_gui
    def load_new_pointcloud(self, project):
        print("Loading all the points... ", project.get_graphUri())
        self.gui.set_statusbar_content("Connecting to server...", 5)
        self.sparql_client = SparqlBackend(project)
        print("Connected to server")
        self.gui.set_statusbar_content("Loading points from server...", 600)
        coords, colors = self.sparql_client.get_all()
        print("Points received from db")
        self.gui.set_statusbar_content("Points loaded", 5)
        self.viewer_client.set(point_size=self.config.get_visualizer_pointsSize())
        self.viewer_client.load(coords, colors)

    def view_point_details(self, id):
        iri = self.sparql_client.get_point_iri(id)
        details = self.sparql_client.get_node_details(iri)
        self.gui.view_node_details(details, iri)

    def view_node_details(self, iri):
        details = self.sparql_client.get_node_details(iri)
        self.gui.view_node_details(details, iri)

    @report_errors_to_gui
    def annotate_node(
            self, subject_iri, predicate_name, object_name_or_value, author_name
    ):
        onto = self.sensorArgs.onto
        predicate = getattr(onto, predicate_name)
        pop_base = self.sensorArgs.populated_base
        subject = owl2.IRIS[subject_iri[1:-1]]
        if predicate in onto.object_properties():
            obj = getattr(onto, object_name_or_value)
        smf.command_manual_annotation(self.sensorArgs, subject, predicate, obj, author_name)

    def select_all_subjects(self, predicate, object):
        selected_colors = self.sparql_client.select_all_subjects(predicate, object)
        self.viewer_client.attributes(self.sparql_client.colors, selected_colors)
        self.viewer_client.set(curr_attribute_id=1)

    @report_errors_to_gui
    def tabular_query(self, query):
        header, content = self.sparql_client.raw_query(query)
        rows = list(map(lambda r: tuple(map(str, r)), content))
        self.gui.plot_tabular(header, rows)

    def _send_legend(self, scalars):
        minimum = float(min(scalars))
        maximum = float(max(scalars))
        step = (maximum - minimum) / NUMBER_OF_LABELS_IN_LEGEND
        # TODO better float format
        labels = [f"{minimum + step * i:.2f}" for i in range(NUMBER_OF_LABELS_IN_LEGEND)]
        colors = get_color_map()
        # it's a numpy array of shape (N, 3), convert to list of hex colors
        colors = ["#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in colors]
        self.gui.set_legend(colors, labels)

    @report_errors_to_gui
    def configure_AWS_connection(
            self, access_key_id, secret_access_key, region, profile_name
    ):
        aws.set_aws_credentials(access_key_id, secret_access_key, region, profile_name)
        self.gui.set_statusbar_content("AWS configured for this device!", 5)

    @report_errors_to_gui
    def add_sensor(
            self,
            sensor_name,
            object_name,
            property_name,
            cert_pem_path,
            private_key_path,
            root_ca_path,
            mqtt_topic,
            client_id,
    ):
        ##### set args
        self.sensorArgs.sensor_name = sensor_name
        self.sensorArgs.object_name = object_name
        self.sensorArgs.property_name = property_name
        self.sensorArgs.cert_pem_path = cert_pem_path
        self.sensorArgs.private_key_path = private_key_path
        self.sensorArgs.root_ca_path = root_ca_path
        self.sensorArgs.mqtttopic = mqtt_topic
        self.sensorArgs.client_id = client_id
        self.gui.set_statusbar_content("Adding Sensor...", 5)
        #####execute function
        smf.command_add_sensor(self.sensorArgs)
        self.gui.set_statusbar_content("Sensor Added!", 5)
        ##### update onto
        self.sensorArgs.onto = owl2.get_ontology(self.sensorArgs.ont_path).load()
        self.gui.set_statusbar_content("Ontology Updated!", 5)
        self.gui.set_statusbar_content(
            "You can add other sensors or update their value, but refresh server connection to see it in the viewer",
            5,
        )

    @report_errors_to_gui
    def update_sensors_and_reason(self):
        self.gui.set_statusbar_content("Updating all Sensors and Reasoning...", 5)
        smf.command_update_sensors_and_reason(self.sensorArgs)
        self.gui.set_statusbar_content("Sensors Updated, Reasoning executed!", 5)
        ##### update onto
        self.sensorArgs.onto = owl2.get_ontology(self.sensorArgs.ont_path).load()
        self.gui.set_statusbar_content("Ontology Updated!", 5)
        self.gui.set_statusbar_content(
            "You can add other sensors or update their value, but refresh server connection to see it in the viewer",
            5,
        )

    ##### ONLY FOR DEBUGGING, UNTIL REAL ARG SETTING IS PREPARED IN "CONNECT TO SERVER" METHOD
    @report_errors_to_gui
    def provisional_set_args(
            self,
            graph_uri,
            ont_path,
            pop_ont_path,
            namespace,
            populated_namespace,
            virtuoso_isql,
    ):

        self.sensorArgs.graph_uri = graph_uri
        self.sensorArgs.ont_path = ont_path
        self.sensorArgs.pop_ont_path = pop_ont_path
        self.sensorArgs.onto = owl2.get_ontology(self.sensorArgs.pop_ont_path).load()
        self.sensorArgs.base = owl2.get_namespace(namespace)
        self.sensorArgs.populated_base = owl2.get_namespace(populated_namespace)
        self.sensorArgs.virtuoso_isql = virtuoso_isql
        import SPARQLWrapper

        wrapper = SPARQLWrapper.Wrapper.SPARQLWrapper(
            endpoint="http://localhost:8890/sparql"
        )
        wrapper.setReturnFormat("csv")
        wrapper.setCredentials("dba", "dba")
        self.sensorArgs.wrapper = wrapper
        self.gui.set_statusbar_content("SensorArgs configured!", 5)

    def natural_language_query(self, nl_query):
        print("Natural language query: ", nl_query)
        onto_path = self.project.get_ontoPath()
        openai_client = init_client()  # TODO understand if can be done only once
        query = nl_2_sparql(nl_query, onto_path, self.project.get_ontologyNamespace(), self.project.get_graphUri(),
                            openai_client, self.gui)
        query = "\n".join(query)
        print("Generated SPARQL query: ", query)
        result, query_type = self.sparql_client.autodetect_query_nl(query)
        if query_type == "tabular":
            header = list(result.keys())
            rows = list(zip(*(result[key] for key in result)))
            self.gui.plot_tabular(header, rows)
        elif query_type == "scalar":
            self.viewer_client.attributes(self.sparql_client.colors, result)
            self.viewer_client.set(curr_attribute_id=1)
            self._send_legend(result)
        elif query_type == "select":
            self.viewer_client.attributes(self.sparql_client.colors, result)
            self.viewer_client.set(curr_attribute_id=1)
        else:
            print("Error, unknown query type: ", query_type)  # TODO remove, shouldn't happen

    def update_project_list(self):
        lst = Project.get_project_list()
        self.gui.set_project_list(lst)

    def open_project(self, project_name):
        print("Opening project:", project_name)
        self.project = Project(project_name)
        self.app_state.set_projectName(self.project.get_name())
        self.gui.set_statusbar_content(f"Opened project: {project_name}", 5)
        self.load_new_pointcloud(self.project)

    def create_project(self, project_name, db_url, graph_uri, graph_namespace, is_local, original_path, onto_namespace):
        print("Creating project: ", project_name)
        if Project.exists(project_name):
            # TODO use proper error handling in GUI
            self.gui.set_query_error(f"Project '{project_name}' already exists!")
            return

        self.project = Project(project_name)
        self.project.set_name(project_name)
        self.project.set_dbUrl(db_url)
        self.project.set_graphUri(graph_uri)
        self.project.set_graphNamespace(graph_namespace)
        self.project.set_ontologyNamespace(onto_namespace)
        self.project.set_isLocal(is_local)
        self.project.set_originalPath(original_path)
        self.project.save()
        self.update_project_list()
        self.gui.set_statusbar_content(f"Created project: {project_name}", 5)
        self.open_project(project_name)  # maybe remove this

    def set_color_scale(self, low, high):
        self.viewer_client.color_map("jet", (low, high))

    def rotate_around(self, n_steps=12):
        theta = self.viewer_client.get('theta')[0]
        distance = self.viewer_client.get('r')[0]
        current_phi = self.viewer_client.get('phi')[0]
        lookat = self.viewer_client.get('lookat')
        poses = [[*lookat, i * pi * 2 / n_steps + current_phi, theta, distance] for i in range(n_steps + 1)]
        self.viewer_client.play(poses, repeat=True)
