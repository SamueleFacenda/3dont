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
      depsPkg = import ./nix; # function that needs to be called with lib and callPackage
      add_py = final: prev: {
        python3 = prev.python3.override {
          packageOverrides = finalPy: prevPy: (final.callPackage depsPkg { }).py;
        };
        python3Packages = final.python3.pkgs;
        qlever-control = final.py.qlever-control; # It's not a library, it's an application
      };
      hide_deps = final: prev: {
        _custom_deps = final.callPackage depsPkg {  };
      };
      # since this overlay needs pkgs even to be evaluated as attribute set, we cannot use final.callPackage
      custom_deps = final: prev: depsPkg { inherit (prev) lib; inherit (final) callPackage; };
    in

    flake-utils.lib.eachDefaultSystem (system:
      let 
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [
            hide_deps
            custom_deps
            add_py
          ];
        };
      in
      {
        packages = rec {
          default = threedont;
          threedont = pkgs.python3.pkgs.buildPythonApplication {
            pname = "threedont";
            src = pkgs.lib.cleanSource ./.;
            inherit version;
            pyproject = true;

            stdenv = pkgs.clang18Stdenv; # better interoperability with darwin build env
            build-system = with pkgs.python3Packages; [
              scikit-build-core
            ];
            
            cmakeBuildType = "Release";
            dontUseCmakeConfigure = true;
            cmakeFlags = pkgs.lib.optionals pkgs.stdenv.hostPlatform.isMacOS [
              "-DADDITIONAL_COMPILER_FLAGS=-fexperimental-library"
            ];

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

            # QLever uses jemalloc, so we preload it here as well (ugly, but works)
            qtWrapperArgs = [ "--set LD_PRELOAD ${pkgs.jemalloc}/lib/libjemalloc.so.2" ];
            
            buildInputs = with pkgs; [
              eigen
              qt6.qtbase
              # hdt
              # qendpoint
              qlever
              fast-float
              # (graalvm-oracle.overrideAttrs {
              #   src = fetchurl {
              #     hash = "sha256-1KsCuhAp5jnwM3T9+RwkLh0NSQeYgOGvGTLqe3xDGDc=";
              #     url = "https://download.oracle.com/graalvm/25/latest/graalvm-jdk-25_linux-x64_bin.tar.gz";
              #   };
              # }) # jre
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
              # jpype1
              psutil
              # oxrdflib
              # pyoxigraph
              scikit-learn
            ];
          };
        } // pkgs._custom_deps; # expose custom deps as packages

        devShells = {
          default = pkgs.mkShell {
            # inherit (self.packages.${system}.threedont) qtWrapperArgs;
            inputsFrom = [ self.packages.${system}.threedont ];
            packages = with pkgs; [
              python3Packages.build
              qt6.qttools
              gdb
              lldb
              qlever-control
              qt6.wrapQtAppsHook
              makeWrapper
            ] ++ lib.optionals stdenv.hostPlatform.isLinux [ gammaray ];
            # Qendpoint config
            JAVA_OPTIONS="-Xmx32G -Dspring.autoconfigure.exclude=org.springframework.boot.autoconfigure.http.client.HttpClientAutoConfiguration -Dspring.devtools.restart.enabled=false";
            # https://discourse.nixos.org/t/python-qt-woes/11808/10
            shellHook = let
              myQtWrapperArgs = pkgs.lib.concatStringsSep " " self.packages.${system}.threedont.qtWrapperArgs;
            in
             ''
              setQtEnvironment=$(mktemp --suffix .setQtEnvironment.sh)
              # echo "shellHook: setQtEnvironment = $setQtEnvironment"
              makeWrapper "/bin/sh" "$setQtEnvironment" "''${qtWrapperArgs[@]}" ${myQtWrapperArgs}
              sed "/^exec/d" -i "$setQtEnvironment"
              source "$setQtEnvironment"
              # cat "$setQtEnvironment"
            '';
          };
        };
      });
}
