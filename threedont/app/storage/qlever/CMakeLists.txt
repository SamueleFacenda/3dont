add_library(pyqlever MODULE
        PyQlever.cpp
        PyQleverQueryResult.cpp
        module.cpp
)

if (WIN32)
    target_compile_options(pyqlever PRIVATE /bigobj)
elseif (UNIX)
    set_target_properties(pyqlever PROPERTIES PREFIX "")
endif ()

# Ugly trick because QLever doesn't configure include directories properly (target_include_directories)
target_include_directories(pyqlever PRIVATE ${Python_INCLUDE_DIRS} ${QLever_DIR}/../../../include/qlever)
target_link_libraries(pyqlever PRIVATE QLever::qlever ${Python_LIBRARIES})

# install in source directory for development
if (THREEDONT_DEVELOP_BUILD)
    # send message
    set_target_properties(pyqlever PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_WITH_INSTALL_RPATH FALSE
    )
    message(STATUS "Installing gui in source directory for development")
    install(TARGETS pyqlever DESTINATION ${CMAKE_SOURCE_DIR}/threedont/app/storage)
else ()
    install(TARGETS pyqlever DESTINATION threedont/app/storage)
endif ()