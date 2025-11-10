# nix comment
{
  description = "3dont, ontology pointcloud visualizer";

  # inputs.nixpkgs.url = "nixpkgs/nixos-25.05";
  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";

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
              version = "0.5.0";
              src = fetchPypi {
                inherit pname version;
                hash = "sha256-+DFI4sbUQ/dxjG6JNsa4njbrtPEALaaejwZW6PtaDfI=";
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
          qlever = pkgs.stdenv.mkDerivation {
            pname = "qlever";
            version = "unstable";
            src = pkgs.fetchFromGitHub {
              repo = "qlever";
              owner = "ad-freiburg";
              rev = "6430af6d4c1298f13ea8f0d47e3d37986fd18263";
              sha256 = "sha256-7vLAD1r7YhEuqMAgLofKBdw29c8RZvb6rwLBeZGSxnY=";
            };
            nativeBuildInputs = with pkgs; [
              cmake
              ninja
            ];
            buildInputs = with pkgs; [
              boost
              icu
              jemalloc
              openssl.dev
              gtest
              nlohmann_json
              antlr
              # range-v3
              zstd
              ctre
              abseil-cpp
              s2geometry
              re2
            ];
            cmakeFlags = [
              "-DUSE_PARALLEL=true"
              # "-D_NO_TIMING_TESTS=ON"
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
