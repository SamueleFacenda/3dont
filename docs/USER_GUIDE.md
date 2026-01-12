# 3DONT User Guide

## Installation
3dont require some dependencies that needs to be built from source and fetched autonomously, in order
to simplify the installation [the nix package manager](https://nixos.org/explore/) is used. It allows
to create fully reproducible builds. It is the only supported packagement method.
- Install nix (package manager): [download](https://nixos.org/download/)
- Enable nix flakes: [guide](https://nixos.wiki/wiki/Flakes#Other_Distros.2C_without_Home-Manager)
- Run `nix run github:SamueleFacenda/3dont`
- Alternatively, clone [https://github.com/SamueleFacenda/3dont.git](https://github.com/SamueleFacenda/3dont.git), enter the
root dir and run `nix run`, or `nix build` and then `./result/bin/threedont` (this way you have the 3dont executable linked in 
`result/bin`).

## Configuration

3dont supports configuration via a config file, the file is located in `~/.config/threedont/` under linux and 
`/Users/<your username>/Library/Application Support/threedont`. It's a simple `.ini` file, like this: 
```
[visualizer]
pointssize = 0.01
scalarcolorscheme = jet
highlightcolor = #FF0000

[general]
loadlastproject = True
```
The config options should be pretty understandable, `highlighcolor` is the color used for results of a select query,
`loadlastproject` let's you choose to open the last project when starting the app, `scalarcolorscheme` is the color
scheme used for scalar queries, you can see the options [here](https://heremaps.github.io/pptk/viewer.html#pptk.viewer.color_map).

## Usage

### Menu
For the sensor stuff ask Matteo.

The view menu allows to choose to view or not the legend after scalar queries and to start a camera animation that rotates around
the view center.

The file menu allows to open a project or to create a new one.
#### Creating a project
There are two types of projects, local and remote projects (Server URL source). The former is hanlded by 3dont while the latter
is relying on an external sparql endpoint.
##### Local projects
You need to specify the path of the original graph, it must be a n-quads or turtle file. 
Then the graph URI is needed (keep the default if not needed), the ontology namespace (the namespace of the base ontology used e.g. 
the 3dontCORE URI) and the graph namespace (the namespace of the populating entities).
##### Remote projects
You need to specify the remote endpoint URL (sparql endpoint), the graph URI (inside the remote endpoint) and the base ontology namespace.

### Querying
There are three query types plus the NL query. These queries must have a complete structure and the result formatted right:
- select: only one column named `p` with the URI of the selected points
- scalar: two columns, `s` and `x`, `s` is the URI of the subject and `x` the scalar value
- tabular: no constrains but the local storage only supports column of the same type (only scalars or only strings)

Scalar queries will show a legend with sliders to adjust the color scale thresholds
### Click-based exploration
Using `ctrl+click` it is  possible to select a point, the point details will be displayed in a dock on the left of the 3d viewer.
Selecting another point will add it to the list of details. The details menu is a simple tree based menu that let's you explore
the attributes of a point and his related entities.
Expanding a `Constitutes` row will highligh the object points. 

It's possible to make some quick queries from the tree menu, right clicking a row will allow three types of actions:
- plot predicate: run a scalar query on the selected predicate and show the results for every point, useful to plot quickly a simple
property
- select all: highligh all the points that has this predicate-object (e.g. all the points that constitutes an object)
- annotate: instert a new triple for that subject with the specified predicate and object.

### Keybindings
A few keybindings are available in the viewer:
- `5`: toggle perspective and orthographic view
- `1,3,7`: align the view to the y,x or z axis
- `left,right,up,down`: rotate around the center
- `[,8,],9`: switch between query result view (highligh or scalar colors) and normal color view.
- `c`: move the center to the center of the selection (or the entire pointcloud if nothing is selected)
- `double click`: set view center to that point
- `ctrl-click`: single point select
- `ctrl-drag`: multiple points select
- `ctrl-shift-drag`: multiple points deselect
- `right click`: deselect all the points
- `wheel`: zoom in/out
