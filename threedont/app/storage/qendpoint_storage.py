from .storage_factory import StorageFactory
from .abstract_storage import AbstractStorage
from .query_handlers import QendpointQuery

import jpype
import jpype.imports
# from jpype.types import *

import os
import shutil
from pathlib import Path
import psutil

def _to_jList(*args):
    from java.util import ArrayList
    jlist = ArrayList()
    for arg in args:
        jlist.add(arg)
    return jlist

@StorageFactory.register(is_local=True, priority=10)
class QendpointStorage(AbstractStorage):
    """
    Storage class that uses Qenpoint to interact with RDF data.
    This class is designed to handle RDF queries and updates using the Qendpoint library.
    """

    def setup_storage(self, identifier: str = None, endpoint: str = None):
        # get path of qendpoint.sh from PATH
        qendpoint_path = shutil.which("qendpoint.sh")
        if qendpoint_path is None:
            raise Exception("qendpoint.sh not found in PATH")

        base = Path(qendpoint_path).parent.parent / 'lib'
        jars = [ os.path.join(base, f) for f in os.listdir(base) if f.endswith('.jar')]

        half_gb = psutil.virtual_memory().total // (1024 ** 3) // 2
        jpype.startJVM(f"-Xmx{half_gb}G", classpath=jars)
        self.base_iri = identifier

    def __del__(self):
        self.graph.shutDown()

    def query(self, query: str, chunked: bool = True):
        return QendpointQuery(self.graph, query)

    def is_empty(self) -> bool:
        return not self.graph.executeBooleanQuery("ASK { { SELECT * WHERE { ?s ?p ?o } LIMIT 1 }}", 10000)

    def bind_to_path(self, path: str):
        from com.the_qa_company.qendpoint.compiler import CompiledSail
        from com.the_qa_company.qendpoint.store import EndpointFiles
        from com.the_qa_company.qendpoint.core.options import HDTOptions, HDTOptionsKeys

        spec = HDTOptions.of(HDTOptionsKeys.LOAD_HDT_TYPE_KEY, HDTOptionsKeys.LOAD_HDT_TYPE_VALUE_MAP,
                             HDTOptionsKeys.BITMAPTRIPLES_INDEX_OTHERS, "spo,ops,pos,osp",
                             HDTOptionsKeys.TEMP_DICTIONARY_IMPL_KEY, HDTOptionsKeys.TEMP_DICTIONARY_IMPL_VALUE_HASH_PSFC,
                             HDTOptionsKeys.HDT_SUPPLIER_KEY, HDTOptionsKeys.LOADER_CATTREE_HDT_SUPPLIER_VALUE_MEMORY,
                             HDTOptionsKeys.ASYNC_DIR_PARSER_KEY, 0,
                             HDTOptionsKeys.DICTIONARY_TYPE_KEY, HDTOptionsKeys.DICTIONARY_TYPE_VALUE_MULTI_OBJECTS_LANG_PREFIXES)

        path = Path(path)
        endpoint_files = EndpointFiles(path / "native-store", path / "hdt-store", "index_dev.hdt")
        repository = CompiledSail.compiler().withEndpointFiles(endpoint_files).withHDTSpec(spec).compileToSparqlRepository()
        repository.init()
        self.graph = repository
        self.path = path

    def _cleanup(self):
        self.graph.shutDown()
        shutil.rmtree(Path(self.path) / "native-store", ignore_errors=True)
        shutil.rmtree(Path(self.path) / "hdt-store", ignore_errors=True)

    # Convert a dictionaryMultiObj to dictionaryMultiObjLangPrefixes
    def _convert_dictionary_type(self):
        from com.the_qa_company.qendpoint.core.tools import HDTConvertTool
        from com.the_qa_company.qendpoint.core.util.listener import ColorTool

        main_file = self.path / "hdt-store" / "index_dev.hdt"
        secondary_file = self.path / "hdt-store" / "index_dev_converted.hdt"

        hdtconvert = HDTConvertTool()
        colorTool = hdtconvert.getClass().getDeclaredField("colorTool")
        colorTool.setAccessible(True)
        colorTool.set(hdtconvert, ColorTool(hdtconvert.color, hdtconvert.quiet))
        hdtconvert.parameters = _to_jList(str(main_file), str(secondary_file), "dictionaryMultiObjLang")
        hdtconvert.execute()

        os.remove(main_file)

        hdtconvert.parameters = _to_jList(str(secondary_file), str(main_file), "dictionaryMultiObjLangPrefixes")
        hdtconvert.execute()

        os.remove(secondary_file)

    def load_file(self, path: str):
        from com.the_qa_company.qendpoint.core.tools import RDF2HDT
        from com.the_qa_company.qendpoint.core.util.listener import ColorTool

        cores = os.cpu_count() or 4
        rdf2hdt = RDF2HDT()
        colorTool = rdf2hdt.getClass().getDeclaredField("colorTool")
        colorTool.setAccessible(True)

        rdf2hdt.color = True
        rdf2hdt.baseURI = self.base_iri
        rdf2hdt.rdfInput = path
        rdf2hdt.hdtOutput = str(self.path / "hdt-store" / "index_dev.hdt")
        rdf2hdt.multiThreadLog = True
        rdf2hdt.generateIndex = True
        rdf2hdt.options = ";".join([
            "autoIndexer.indexName=index_dev",
            "dictionary.type=dictionaryMultiObj",
            "bitmaptriples.indexmethod=recommended",
            "bitmaptriples.index.others=spo,ops,pos,osp",
            "loader.type=disk",
            "loader.disk.chunkSize=10000000",
            "loader.disk.compressMode=compressionComplete",
            f"loader.disk.compressWorker={cores}",
            "loader.disk.fileBufferSize=2097152",
            "loader.disk.maxFileOpen=1024",
            "tempDictionary.impl=hashPsfc",
            f"loader.cattree.kcat={cores}",
            "profiler=false",
        ])
        colorTool.set(rdf2hdt, ColorTool(rdf2hdt.color, rdf2hdt.quiet))

        self._cleanup() # Clean up existing data before loading new data
        (self.path / "hdt-store").mkdir()

        rdf2hdt.execute()
        self._convert_dictionary_type()

        self.bind_to_path(str(self.path)) # Reopen the graph after loading new data

    def update(self, query: str):
        self.graph.executeUpdate(query, 10000, None)
