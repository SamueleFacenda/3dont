{ clang18Stdenv
, lib
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
let
  stdenv = clang18Stdenv;
in
stdenv.mkDerivation rec {
  pname = "qlever";
  version = "unstable";
  src = fetchFromGitHub {
    repo = "qlever";
    owner = "ad-freiburg";
    # rev = "ae21741ece88a14bd8392e3167bd84f89c25969d";
    # hash = "sha256-56/33GR+t6TFO/cWJNk7ZbL8v/BXH+q7Qkhy11QppNE=";
    rev = "5b09c6a55da48b0d7d9bc958f2ca3458d07fd9e0";
    hash = "sha256-f/q36+8/26e75VUrQdqjGgmpv9KEE7hlC8fapeHxMNw=";
  };
  patches = [ ./qlever.patch ];
  cmakeBuildType = "Release";
  cmakeFlags = [
    "-DUSE_PARALLEL=${if stdenv.hostPlatform.isLinux then "true" else "false"}"
    "-DSINGLE_TEST_BINARY=On"
    "-DADDITIONAL_COMPILER_FLAGS=-fsized-deallocation" # needed by clang 18
  ] ++ lib.optionals stdenv.hostPlatform.isMacOS [
    "-DADDITIONAL_COMPILER_FLAGS=-fexperimental-library"
  ];
  buildFlags = [ "qlever" ]; # build only libqlever target!
  installPhase = "cmake --install . -j \${NIX_BUILD_CORES}"; # the default (make install) tries to build all the tests and fails.
  nativeBuildInputs = [
    cmake
    pkg-config
  ];
  propagatedBuildInputs = [ # propagate everything since there are no private dependencies
    boost
    icu
    (jemalloc.override { disableInitExecTls = true; })
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
    (re2.override {abseil-cpp = abseil-cpp.override {cxxStandard = "20";};}).dev
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
}
