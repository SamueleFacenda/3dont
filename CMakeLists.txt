cmake_minimum_required(VERSION 3.15)
project(threedont)

# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 20)

option(THREEDONT_DEVELOP_BUILD "Enable development build" OFF)

find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Qt6 CONFIG REQUIRED COMPONENTS Widgets Network Gui OpenGLWidgets Core)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(threedont)
