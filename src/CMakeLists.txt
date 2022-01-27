cmake_minimum_required (VERSION 3.10)

project ("neuronephys" VERSION 1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(target neuronephys)
#Sources
set(source_path ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE cpp_sources ${source_path}/*.cpp)
file(GLOB_RECURSE cu_sources ${source_path}/*.cu EXCLUDE ${source_path}/main.cu)
#External libraries:
list(APPEND CMAKE_MODULE_PATH ${source_path})
find_package(ANN REQUIRED)
find_package(HDF5 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(ArrayFire REQUIRED)
find_package(GLM REQUIRED)#Header only
add_executable(${target} main.cu ${cu_sources} ${cpp_sources})
target_include_directories(${target} PUBLIC 
${ArrayFire_INCLUDE_DIRS}
${HDF5_INCLUDE_DIRS}
${GLEW_INCLUDE_DIRS}
${GLFW_INCLUDE_DIRS}
${GLM_INCLUDE_DIRS}
${ANN_INCLUDE_DIRS}
)
target_link_libraries(${target}
GLEW::GLEW
glfw
HDF5
ArrayFire::afcuda
${ANN_LIBRARIES}
)