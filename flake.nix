# nix comment
{
  description = "3dont, ontology pointcloud visualizer";

  # inputs.nixpkgs.url = "nixpkgs/nixos-25.05";
  # inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  inputs.nixpkgs.url = "nixpkgs/dbeacf1";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    let
      version = "0.0.1";
      overlay = final: prev: {
        python3 = prev.python3.override {
          packageOverrides = finalPy: prevPy: {};
        };
        python3Packages = final.python3.pkgs;
      };
    in

    flake-utils.lib.eachDefaultSystem (system:
      let 
        pkgs = ((import nixpkgs { inherit system; config.allowUnfree = true; }).extend overlay);
      in
      {

        packages = rec {
          default = threedont;
          threedont = pkgs.python3.pkgs.buildPythonApplication {
            pname = "threedont";
            src = pkgs.lib.cleanSource ./.;
            inherit version;
            pyproject = true;

            stdenv = pkgs.clangStdenv; # better interoperability with darwin build env

#            cmakeFlags = [
#              "-DCMAKE_BUILD_TYPE=Debug"
#            ];
            
            build-system = with pkgs.python3Packages; [
              scikit-build-core
            ];
            
            dontUseCmakeConfigure = true;

            nativeBuildInputs = with pkgs; [
              qt6.wrapQtAppsHook
              pkg-config
              cmake
              ninja
            ];

            dontWrapQtApps = true;
            # See note here: https://nixos.org/manual/nixpkgs/unstable/#qt-runtime-dependencies
            postFixup = ''
                wrapQtApp "$out/bin/threedont"
            '';
            
            buildInputs = with pkgs; [
              eigen
              qt6.qtbase
              hdt
              qendpoint
              qlever
              (graalvm-oracle.overrideAttrs {
                src = fetchurl {
                  hash = "sha256-1KsCuhAp5jnwM3T9+RwkLh0NSQeYgOGvGTLqe3xDGDc=";
                  url = "https://download.oracle.com/graalvm/25/latest/graalvm-jdk-25_linux-x64_bin.tar.gz";
                };
              }) # jre
            ] ++ lib.optionals stdenv.hostPlatform.isLinux [
              libGL
              qt6Packages.qtstyleplugin-kvantum
            ];
            
            dependencies = with pkgs.python3Packages; [
              numpy
              sparqlwrapper
              rdflib
              networkx
              openai
              owlready2
              platformdirs
              nltk
              editdistance
              jarowinkler
              boto3
              awsiotpythonsdk
              oxrdflib
              pyoxigraph
              jpype1
              psutil
            ];
          };
          owlready2 = pkgs.python3Packages.callPackage ({buildPythonPackage, fetchPypi, distutils, setuptools, cython }:
            buildPythonPackage rec {
              pname = "owlready2";
              version = "0.47";
              pyproject = true;
              build-system = [ setuptools ];
              src = fetchPypi {
                inherit pname version;
                hash = "sha256-r34dIgXAtYhtLjQ5erjBDKKf9ow9w3AtQzk5Zqx/brA=";
              };
              dependencies = [
                distutils
                cython
              ];
            }
          ) {};
          oxrdflib = pkgs.python3Packages.callPackage ({buildPythonPackage, fetchPypi, pyoxigraph, rdflib, setuptools, setuptools-scm, uv-build}:
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
          ) {};
          qendpoint = pkgs.stdenv.mkDerivation rec {
            pname = "qendpoint-cli";
            version = "2.5.2";

            src = pkgs.fetchurl {
              url = "https://github.com/the-qa-company/qEndpoint/releases/download/v${version}/qendpoint-cli.zip";
              hash = "sha256-9PIDVBOUBe6AMVvKUMWL9otrJLKTWcuDBkpHvs5cHPw=";
            };

            buildInputs = [ pkgs.jre ];

            nativeBuildInputs = with pkgs; [ makeWrapper unzip ];

            installPhase = ''
              cp -r . "$out"

              rm "$out"/bin/*.bat
              rm "$out"/bin/*.ps1
              sed -i 's/javaenv\.sh/qendpoint-javaenv.sh/g' $(ls "$out"/bin/*.sh | grep -v javaenv);
              mv "$out"/bin/javaenv.sh "$out"/bin/qendpoint-javaenv.sh

              for i in $(
                  find "$out"/bin \
                    -type f -name "*.sh" \
                    \! -regex '.*\/\(qendpoint\|qep\|[^/]*javaenv\)[^/]*\.sh$' \
                    -printf '%f\n' \
                ); do
                i_quotemeta="$(printf '%s\n' "$i" | sed -e 's/[.[\*^$/]/\\&/g')"
                sed -i 's,bin/'"$i_quotemeta"',bin/qendpoint-'"$i"',g' $(ls "$out"/bin/*.sh);
                mv "$out"/bin/$i "$out"/bin/qendpoint-$i
              done

              for i in $(ls "$out"/bin/*.sh | grep -v javaenv); do
                wrapProgram "$i" --prefix "PATH" : "${pkgs.jre}/bin/" \
                  --set-default JAVA_OPTIONS "-Dspring.autoconfigure.exclude=org.springframework.boot.autoconfigure.http.client.HttpClientAutoConfiguration -Dspring.devtools.restart.enabled=false"
              done
            '';
          };
          qlever-control = pkgs.python3Packages.buildPythonApplication {
            pname = "qlever-control";
            version = "unstable";
            src = pkgs.fetchFromGitHub {
              repo = "qlever-control";
              owner = "qlever-dev";
              rev = "8af04bb0f74b7327b38a138a13d3aaa6f96e7f3c";
              sha256 = "sha256-Swnwng6zFbqXPSfuaRHso8FeCh2vuWUG61j0VPpN9LQ=";
            };
            pyproject = true;
            build-system = with pkgs.python3Packages; [ setuptools ];
            dependencies = with pkgs.python3Packages; [
              psutil
              termcolor
              argcomplete
              pyyaml
              rdflib
              (buildPythonPackage rec {
                pname = "requests_sse";
                version = "0.5.2";
                pyproject = true;
                build-system = [ poetry-core ];
                dependencies = [ requests ];
                src = fetchPypi {
                  inherit version pname;
                  sha256 = "sha256-K8t8+QUHSxj/n3MicWI0wRiN/egFu6ODALN8a1rjogo=";
                };
              })
            ];
          };
          qlever = pkgs.stdenv.mkDerivation {
            pname = "qlever";
            version = "unstable";
            src = pkgs.fetchFromGitHub {
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
            buildFlags = [ "qlever" ];
            installPhase = "cmake --install . -j \${NIX_BUILD_CORES}"; # the default (make install) tries to build all the tests and fails.
            nativeBuildInputs = with pkgs; [
              cmake
              pkg-config
            ];
            buildInputs = with pkgs; [
              boost
              icu
              jemalloc
              openssl.dev
              nlohmann_json
              (range-v3.overrideAttrs { src = pkgs.fetchFromGitHub {
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
              ((gtest.overrideAttrs { src = pkgs.fetchFromGitHub {
                owner = "google";
                repo = "googletest";
                rev = "7917641ff965959afae189afb5f052524395525c";
                hash = "sha256-Pfkx/hgtqryPz3wI0jpZwlRRco0s2FLcvUX1EgTGFIw=";
              };}).override {cxx_standard = "20";}).dev
              antlr4.runtime.cpp.dev
              (pkgs.stdenv.mkDerivation {
                pname = "fsst";
                version = "unstable";
                src = pkgs.fetchFromGitHub {
                  repo = "fsst";
                  owner = "cwida";
                  rev = "89f49c580c6388acf3b6ed2a49e1bfde6c05e616";
                  sha256 = "sha256-kP9InvstinqmOnC2X8pPad7t8h79W2VpUnDlW0F9ELQ=";
                };
                nativeBuildInputs = with pkgs; [ cmake ];
                installPhase = ''
                  mkdir -p $out/lib
                  mv libfsst.a libfsst12.a $out/lib
                  mkdir -p $out/include/fsst
                  mv ../*.h $out/include/fsst/
                '';
              })
              (pkgs.stdenv.mkDerivation {
                pname = "spatialjoin";
                version = "unstable";
                src = pkgs.fetchFromGitHub {
                    owner = "ad-freiburg";
                    repo = "spatialjoin";
                    rev = "7f5c9007090cd1b902ad1873beb064820caebf39";
                    hash = "sha256-zvZCRJByvykVCQM9z0g0vUj1sDHaqhMDTbRzudp7jxQ=";
                    fetchSubmodules = true;
                };
                nativeBuildInputs = with pkgs; [ cmake ];
                patchPhase = ''
                    sed -i 's|enable_testing()|# enable_testing()|g' CMakeLists.txt
                    cat >> src/spatialjoin/CMakeLists.txt << 'EOF'
                    install(
                        TARGETS spatialjoin-dev
                        EXPORT spatialjoin-dev-targets
                        ARCHIVE DESTINATION lib
                        LIBRARY DESTINATION lib
                        RUNTIME DESTINATION bin
                    )
                    install(
                        DIRECTORY ''${SPATIALJOIN_INCLUDE_DIR}/
                        DESTINATION include
                    )
                    install(
                        EXPORT spatialjoin-dev-targets
                        FILE spatialjoin-dev-config.cmake
                        NAMESPACE spatialjoin::
                        DESTINATION lib/cmake/spatialjoin-dev
                    )
                    install(TARGETS pb_util
                            EXPORT pb_utilTargets
                            LIBRARY DESTINATION lib
                            ARCHIVE DESTINATION lib
                            RUNTIME DESTINATION bin
                            INCLUDES DESTINATION include)
                    install(DIRECTORY ''${CMAKE_CURRENT_SOURCE_DIR}/../
                            DESTINATION include
                            FILES_MATCHING PATTERN "*.h")
                    install(EXPORT pb_utilTargets
                            FILE pb_utilConfig.cmake
                            NAMESPACE pb_util::
                            DESTINATION lib/cmake/pb_util)
                    EOF
                '';
              })
            ];
          };
        };

        devShells = {
          default = pkgs.mkShell {
            inputsFrom = [ self.packages.${system}.threedont ];
            packages = with pkgs; [
              python3Packages.build
              qt6.qttools
              gdb
              lldb
              self.packages.${system}.qlever-control
            ] ++ lib.optionals stdenv.hostPlatform.isLinux [ gammaray ];
            nativeBuildInputs = with pkgs; [
              qt6.wrapQtAppsHook
              makeWrapper
            ];
            # Qendpoint config
            JAVA_OPTIONS="-Xmx32G -Dspring.autoconfigure.exclude=org.springframework.boot.autoconfigure.http.client.HttpClientAutoConfiguration -Dspring.devtools.restart.enabled=false";
            # https://discourse.nixos.org/t/python-qt-woes/11808/10
            shellHook = ''
              setQtEnvironment=$(mktemp --suffix .setQtEnvironment.sh)
              # echo "shellHook: setQtEnvironment = $setQtEnvironment"
              makeWrapper "/bin/sh" "$setQtEnvironment" "''${qtWrapperArgs[@]}"
              sed "/^exec/d" -i "$setQtEnvironment"
              source "$setQtEnvironment"
              # cat "$setQtEnvironment"
            '';
          };
        };
      });
}
