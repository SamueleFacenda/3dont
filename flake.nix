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
        pkgs = (nixpkgs.legacyPackages.${system}.extend overlay);
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
              apache-jena
              hdt
              hdt-java
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
          hdt-java = pkgs.stdenv.mkDerivation rec {
              pname = "hdt-java";
              version = "3.0.10";
    
              src = pkgs.fetchurl {
                url = "https://github.com/rdfhdt/hdt-java/releases/download/v${version}/rdfhdt.tar.gz";
                hash = "sha256-m5fyjr6R+9Zkkts202Uxr0BTala9eGyLuYz3CuepeMU=";
              };
    
              buildInputs = [ pkgs.jre ];
    
              nativeBuildInputs = [ pkgs.makeWrapper ];
    
              installPhase = ''
                cp -r . "$out"
    
                rm "$out"/bin/*.bat
                rm "$out"/bin/*.ps1
                sed -i 's/javaenv\.sh/hdt-java-javaenv.sh/g' $(ls "$out"/bin/*.sh | grep -v javaenv);
                mv "$out"/bin/javaenv.sh "$out"/bin/hdt-java-javaenv.sh
                for i in $(ls "$out"/bin/*.sh | grep -v javaenv); do
                  wrapProgram "$i" --prefix "PATH" : "${pkgs.jre}/bin/"
                done
              '';
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
