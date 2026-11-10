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
    material.cpp
    water_material.cpp
    rt_material.cpp
    brute_force_material.cpp
    benchmark.cpp
    vk_utils.cpp
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_vulkan.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/imgui
    ${CMAKE_SOURCE_DIR}/imgui/backends
)

# Benchmark results are appended here so every optimization step is comparable
target_compile_definitions(${PROJECT_NAME} PRIVATE
    BENCHMARK_CSV_PATH="${CMAKE_SOURCE_DIR}/benchmarks.csv"
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

# Shader compilation target
find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/Bin")
if(GLSLC)
    add_custom_target(shaders ALL
        COMMAND ${GLSLC} --target-spv=spv1.5 ${CMAKE_SOURCE_DIR}/shaders/shader.vert -o ${CMAKE_SOURCE_DIR}/shaders/vert.spv
        COMMAND ${GLSLC} --target-spv=spv1.5 ${CMAKE_SOURCE_DIR}/shaders/shader.frag -o ${CMAKE_SOURCE_DIR}/shaders/frag.spv
        COMMAND ${GLSLC} --target-spv=spv1.5 ${CMAKE_SOURCE_DIR}/shaders/shader.mesh -o ${CMAKE_SOURCE_DIR}/shaders/mesh.spv
        COMMAND ${GLSLC} --target-spv=spv1.5 ${CMAKE_SOURCE_DIR}/shaders/shader.task -o ${CMAKE_SOURCE_DIR}/shaders/task.spv
        COMMAND ${GLSLC} --target-spv=spv1.5 --target-env=vulkan1.2 ${CMAKE_SOURCE_DIR}/shaders/raygen.rgen -o ${CMAKE_SOURCE_DIR}/shaders/raygen.spv
        COMMAND ${GLSLC} --target-spv=spv1.5 --target-env=vulkan1.2 ${CMAKE_SOURCE_DIR}/shaders/miss.rmiss -o ${CMAKE_SOURCE_DIR}/shaders/miss.spv
        COMMAND ${GLSLC} --target-spv=spv1.5 --target-env=vulkan1.2 ${CMAKE_SOURCE_DIR}/shaders/closesthit.rchit -o ${CMAKE_SOURCE_DIR}/shaders/closesthit.spv
        COMMAND ${GLSLC} --target-spv=spv1.5 ${CMAKE_SOURCE_DIR}/shaders/brute_force.comp -o ${CMAKE_SOURCE_DIR}/shaders/brute_force.spv
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/shaders
        COMMENT "Compiling shaders..."
    )
    add_dependencies(${PROJECT_NAME} shaders)
endif()
