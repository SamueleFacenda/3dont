{ buildPythonPackage
, fetchPypi
, pyoxigraph
, rdflib
, setuptools
, setuptools-scm
, uv-build
}:
buildPythonPackage rec {
  pname = "oxrdflib";
  # version = "0.5.0";
  version = "0.4.0";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-N9TAJdTjnF5UclJ8OTmYv9EOWsGFow4IC1tRD23X2oY=";
    # hash = "sha256-+DFI4sbUQ/dxjG6JNsa4njbrtPEALaaejwZW6PtaDfI=";
  };
  pyproject = true;
  build-system = [ setuptools setuptools-scm uv-build ];
  dependencies = [
    pyoxigraph
    rdflib
  ];
}
