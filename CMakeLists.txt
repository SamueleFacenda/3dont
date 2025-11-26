cmake_minimum_required(VERSION 3.15)
project(threedont)

# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 20)

option(THREEDONT_DEVELOP_BUILD "Enable development build" OFF)

find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Qt6 CONFIG REQUIRED COMPONENTS Widgets Network Gui OpenGLWidgets Core)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# TODO add  these deps to qlever config
find_package(absl REQUIRED)
find_package(Boost 1.81 COMPONENTS iostreams program_options url container REQUIRED CONFIG)
find_package(OpenSSL REQUIRED)
find_package(re2 REQUIRED)
find_package(ctre REQUIRED)
find_package(ICU 60 REQUIRED COMPONENTS uc i18n)
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(antlr4-runtime REQUIRED)
find_package(range-v3 REQUIRED)
find_package(spatialjoin-dev REQUIRED)
find_package(pb_util REQUIRED)
add_compile_definitions(SPATIALJOIN_NO_BZIP2=True SPATIALJOIN_NO_ZLIB=True)
find_package(PkgConfig)
pkg_check_modules(JEMALLOC jemalloc)
find_path(fsst_INCLUDE_DIR
        NAMES fsst/fsst.h
        REQUIRED
)
set(fsst_INCLUDE_DIR "${fsst_INCLUDE_DIR}/fsst")
find_library(fsst_LIB
        NAMES fsst
        REQUIRED
)
add_library(fsst STATIC IMPORTED)
set_target_properties(fsst PROPERTIES
        IMPORTED_LOCATION "${fsst_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${fsst_INCLUDE_DIR}"
)



find_package(QLever REQUIRED)

add_subdirectory(threedont)
