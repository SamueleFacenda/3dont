{ stdenv
, cmake
, fetchFromGitHub
}:

stdenv.mkDerivation {
  pname = "fsst";
  version = "unstable";
  src = fetchFromGitHub {
    repo = "fsst";
    owner = "cwida";
    rev = "89f49c580c6388acf3b6ed2a49e1bfde6c05e616";
    sha256 = "sha256-kP9InvstinqmOnC2X8pPad7t8h79W2VpUnDlW0F9ELQ=";
  };
  nativeBuildInputs = [ cmake ];
  installPhase = ''
    mkdir -p $out/lib
    mv libfsst.a libfsst12.a $out/lib
    mkdir -p $out/include/fsst
    mv ../*.h $out/include/fsst/
  '';
}
