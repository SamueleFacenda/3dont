# 3DONT viewer

View, query and manually annotate pointclouds ontologies.

### TODO
- [x] select query box
- [x] natural language query box
- [x] scalar query box
- [x] click to view properties of a point
- [x] tree view for details
- [x] check if points are selected right
- [x] manual annotation tool
- [x] scalar field query
- [x] right click: select all subjects with this predicate and object
- [x] right click: plot predicate (scalar)
- [x] append OFFSET and LIMIT instead of using template
- [x] error box in gui
- [x] error handling for: wrong output format, empty output, syntax error, connection error...
- [x] highlight subject in treeview
- [x] non visual queries (tabular result box?)
- [x] highlight subject name
- [x] legend scalar
- [x] adjust licensing
- [x] persistent configuration (platformdirs)
- [x] nl queries
- [x] select min max scalar
- [x] block dock floating
- [x] fix viewer focus issue
- [x] fix OpenGL
- [x] fix some random crashes
- [x] optimize query speed
- [x] make quendpoint/sparql storage superclass
- [x] check [qlever](https://github.com/ad-freiburg/qlever)
- [x] qlever errors handling
- [x] investigate big point glitch
- [x] avoid startup lag for dependency dowload
- [ ] scalar templates
- [ ] refactor input dialogs
- [ ] import pointclouds etc.
- [ ] add default to scalar queries equal to color
- [ ] configure page (namespace)
- [ ] better loading screen
- [ ] gui error report (generic)
- [ ] override config with cli arguments
- [ ] investigate empty view on heritage loading
- [ ] more configuration options (e.g. select color) and settings dialog
- [ ] fix lookat lines width
- [ ] use QOpenglBuffer for buffer management
- [ ] investigate projects not showing sometimes
- [ ] test queries with type:point
- [ ] plot 3d graph with relations
- [ ] check [this](https://github.com/csse-uoft/owlready2/blob/master/src/owlready2/namespace.py) owlready alternative
- [ ] check [rdf-fusion](https://github.com/tobixdev/rdf-fusion)
- [ ] move from QTcpServer to QLocalServer
- [ ] dynamically compute fast rendering LOD (based on render time)

### Tested backends:
- virtuoso
- virtuoso hybrid (points as relational table linked view)
- oxigraph
- owlready2
- rdflib
- oxigraph-rdflib
- qendpoint (with jpype)
- qlever (the best one)

### Backends to test
- rdf-fusion

note
file las pointcloud in project
epsg code in project
owlready2 sqlite database
metodo get_onto o get_graph che ritorna owlready (in project)
output dir in project (con sottocartelle fisse)

## License

The LICENSE_PPTK file is the license of the viewer used as a base for this project. It is relative to the content of the folder
`threedont/gui/viewer` and the file `threedont/app/viewer.py`.

## Install

You have to build the wheel from source.

```bash
python -m build --no-isolation --wheel
pip install .
```

## Build

We provide CMake scripts for automating most of the build process, just 
follow the standard cmake workflow to build:

```bash
mkdir build
cd build
cmake ..
make -j4
```

Set `-DTHREEDONT_DEVELOP_BUILD=On` to have the python modules placed in the source tree. 
This is useful when developing since allows to run the app with `python -m threedont`.

##### Requirements

Listed are versions of libraries used to develop 3dont, though earlier versions
of these libraries may also work.

* [QT](https://www.qt.io/) 6.9
* [Eigen](http://eigen.tuxfamily.org) 3.2.9
* [Python](https://www.python.org/) 3.6+
* [Numpy](http://www.numpy.org/) 1.13
* fast-float 

## Build with nix

This project has nix support (with flakes). Install nix and enable flakes,
the run `nix build` from the root to build the project. 
You can also use `nix run` and `nix run github:SamueleFacenda/3dont` to run the project 
without downloading the repo.

