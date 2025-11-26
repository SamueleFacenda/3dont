{ stdenv
, fetchFromGitHub
, cmake
, pkg-config
, boost
, icu
, jemalloc
, openssl
, nlohmann_json
, range-v3
, zstd
, ctre
, abseil-cpp
, gtest
, re2
, s2geometry
, antlr4
, fsst
, spatialjoin
}:

stdenv.mkDerivation rec {
  pname = "qlever";
  version = "unstable";
  src = fetchFromGitHub {
    repo = "qlever";
    owner = "ad-freiburg";
    rev = "ae21741ece88a14bd8392e3167bd84f89c25969d";
    hash = "sha256-56/33GR+t6TFO/cWJNk7ZbL8v/BXH+q7Qkhy11QppNE=";
  };
  patches = [ ./qlever.patch ];
  cmakeBuildType = "Release";
  cmakeFlags = [
    "-DUSE_PARALLEL=true"
    "-DUSE_CPP_17_BACKPORTS=On"
    "-DSINGLE_TEST_BINARY=On"
  ];
  buildFlags = [ "qlever" ]; # build only libqlever target!
  installPhase = "cmake --install . -j \${NIX_BUILD_CORES}"; # the default (make install) tries to build all the tests and fails.
  nativeBuildInputs = [
    cmake
    pkg-config
  ];
  buildInputs = [
    boost
    icu
    jemalloc
    openssl.dev
    nlohmann_json
    (range-v3.overrideAttrs { src = fetchFromGitHub {
      owner = "joka921";
      repo = "range-v3";
      rev = "5ae161451ec1baaac352d7567298d3ac143bccae";
      hash = "sha256-r7wxSE8dX3hCxtHn8bwAOc3hM8Eodi9V28HaT53AUH8=";
    };})
    zstd
    ctre
    (abseil-cpp.override {cxxStandard = "20";})
    s2geometry
    re2
    ((gtest.overrideAttrs { src = fetchFromGitHub {
      owner = "google";
      repo = "googletest";
      rev = "7917641ff965959afae189afb5f052524395525c";
      hash = "sha256-Pfkx/hgtqryPz3wI0jpZwlRRco0s2FLcvUX1EgTGFIw=";
    };}).override {cxx_standard = "20";}).dev
    antlr4.runtime.cpp.dev
    fsst
    spatialjoin
  ];

  propagatedBuildInputs = buildInputs; # propagate everything since there is not private dependency
}
