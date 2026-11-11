cmake_minimum_required(VERSION 3.20)
project(VKRenderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan SDK (uses VULKAN_SDK environment variable)
find_package(Vulkan REQUIRED)

# GLFW - hardcoded path for this system
set(GLFW_INCLUDE_DIR "C:/Users/musta/OneDrive/Documents/Visual Studio 2022/Libraries/glfw-3.4.bin.WIN64/include")
set(GLFW_LIBRARY "C:/Users/musta/OneDrive/Documents/Visual Studio 2022/Libraries/glfw-3.4.bin.WIN64/lib-vc2022/glfw3.lib")

add_library(glfw INTERFACE IMPORTED)
set_target_properties(glfw PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${GLFW_LIBRARY}"
)

# GLM - header only, hardcoded path
set(GLM_INCLUDE_DIR "C:/Users/musta/OneDrive/Documents/Visual Studio 2022/Libraries/glm")
add_library(glm INTERFACE IMPORTED)
set_target_properties(glm PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIR}"
)

# Source files
set(SOURCES
    main.cpp
    engine.cpp
    volkan.cpp
    rt_material.cpp
    grid_material.cpp
    grid.cpp
    triangle_scene.cpp
    third_party/ufbx/ufbx.c
    benchmark.cpp
    vk_utils.cpp
    window.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/ufbx
)

# ufbx (FBX loader, MIT / public domain): third-party code, so don't show its warnings
set_source_files_properties(third_party/ufbx/ufbx.c PROPERTIES COMPILE_OPTIONS "/w")

target_compile_definitions(${PROJECT_NAME} PRIVATE
    # Library configuration macros change what headers compile to, so they must
    # be identical in every source file. Defining them here guarantees that.
    GLFW_INCLUDE_VULKAN
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS

    # Benchmark results are appended here so every optimization step is comparable
    BENCHMARK_CSV_PATH="${CMAKE_SOURCE_DIR}/benchmarks.csv"
    # Debug/Release, recorded with every result
    BUILD_CONFIG="$<CONFIG>"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    glm
)

# Copy shaders to build directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders $<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders
)

# Shader compilation target. -g embeds the GLSL source and line info in the SPIR-V
# so profilers (Nsight shader profiler) can map GPU time back to shader lines.
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(GLSLC)
    add_custom_target(shaders ALL
        COMMAND ${GLSLC} -g --target-spv=spv1.5 --target-env=vulkan1.2 ${CMAKE_SOURCE_DIR}/shaders/raygen.rgen -o ${CMAKE_SOURCE_DIR}/shaders/raygen.spv
        COMMAND ${GLSLC} -g --target-spv=spv1.5 --target-env=vulkan1.2 ${CMAKE_SOURCE_DIR}/shaders/miss.rmiss -o ${CMAKE_SOURCE_DIR}/shaders/miss.spv
        COMMAND ${GLSLC} -g --target-spv=spv1.5 --target-env=vulkan1.2 ${CMAKE_SOURCE_DIR}/shaders/closesthit.rchit -o ${CMAKE_SOURCE_DIR}/shaders/closesthit.spv
        COMMAND ${GLSLC} -g --target-spv=spv1.5 ${CMAKE_SOURCE_DIR}/shaders/grid_traversal.comp -o ${CMAKE_SOURCE_DIR}/shaders/grid_traversal.spv
        COMMAND ${GLSLC} -g --target-spv=spv1.5 ${CMAKE_SOURCE_DIR}/shaders/triangle_traversal.comp -o ${CMAKE_SOURCE_DIR}/shaders/triangle_traversal.spv
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/shaders
        COMMENT "Compiling shaders..."
    )
    add_dependencies(${PROJECT_NAME} shaders)
endif()
