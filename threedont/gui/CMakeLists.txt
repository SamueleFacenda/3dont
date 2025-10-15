set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)

add_library(gui MODULE
        gui_wrapper.cpp
        controller_wrapper.cpp
        main_layout.cpp
        points_tree_view/graph_tree_model.cpp
        points_tree_view/graph_tree_item.cpp
        widgets/scale_legend.cpp
        viewer/utils/alternative_frame_buffer.cpp
        viewer/data/point_attributes.cpp
        viewer/data/octree.cpp
        viewer/camera/splines.cpp
        viewer/utils/opengl_funcs.cpp
        viewer/camera/qt_camera.cpp
        viewer/components/text.cpp
        viewer/viewer.cpp
        viewer/components/floor_grid.cpp
        viewer/components/background.cpp
        viewer/camera/camera.cpp
        viewer/camera/camera_dolly.cpp
        viewer/components/look_at.cpp
        viewer/components/point_cloud.cpp
        viewer/utils/comm_funcs.cpp
        viewer/components/selection_box.cpp
        dialogs/create_project_dialog.cpp
)

target_sources(gui PRIVATE
        viewer/viewer.h
        main_layout.ui
        main_layout.h
        controller_wrapper.h
        widgets/scale_legend.h
        points_tree_view/graph_tree_model.h
        points_tree_view/graph_tree_item.h
        dialogs/properties_mapping_selection.h
        dialogs/create_project_dialog.h
)

if (WIN32)
    target_compile_options(gui PRIVATE /bigobj)
elseif (UNIX)
    set_target_properties(gui PROPERTIES PREFIX "")
endif ()

target_include_directories(gui PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(gui Qt6::Widgets Qt6::Network Qt6::Gui Qt6::OpenGLWidgets Eigen3::Eigen ${Python_LIBRARIES})

# install in source directory for development
if (THREEDONT_DEVELOP_BUILD)
    # send message
    set_target_properties(gui PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_WITH_INSTALL_RPATH FALSE
    )
    message(STATUS "Installing gui in source directory for development")
    install(TARGETS gui DESTINATION ${CMAKE_SOURCE_DIR}/threedont/gui)
else ()
    install(TARGETS gui DESTINATION threedont/gui)
endif ()
