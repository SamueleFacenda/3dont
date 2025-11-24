# import all nix files in the current folder, and callPackage on them
# The return value is a list of all execution results, which is the list of overlays

{ lib, callPackage}:
# execute and import all packages files in the current directory with the given args
let
  inherit (lib.attrsets) mapAttrs' filterAttrs;
  inherit (builtins) readDir;
  inherit (lib.strings) removeSuffix;
in
mapAttrs'
  (n: v: rec {
    # derivation pname or file/directory name
    name = removeSuffix ".nix" n;
    value = callPackage (./. + "/${n}") { };
  })
  (filterAttrs
    (n: v: n != "default.nix")
    (readDir ./.))
