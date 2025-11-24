{ stdenv
, cmake
, fetchFromGitHub
}:

stdenv.mkDerivation {
  pname = "spatialjoin";
  version = "unstable";
  src = fetchFromGitHub {
      owner = "ad-freiburg";
      repo = "spatialjoin";
      rev = "7f5c9007090cd1b902ad1873beb064820caebf39";
      hash = "sha256-zvZCRJByvykVCQM9z0g0vUj1sDHaqhMDTbRzudp7jxQ=";
      fetchSubmodules = true;
  };
  nativeBuildInputs = [ cmake ];
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
}
