cmake_minimum_required (VERSION 3.10)

project ("ANN" VERSION 1.1)
set(CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (target ANN)
# Sources
set (include_path ${CMAKE_CURRENT_SOURCE_DIR}/include/${target})
set (source_path ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB_RECURSE intermediate_headers ${source_path}/*.h)
file(GLOB_RECURSE sources ${source_path}/*.cpp)
#Build options for DLL
include (GenerateExportHeader)
#Build library
add_library(${target} STATIC ${intermediate_headers} ${sources})
# Set ouput directory
set_target_properties (${target} PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib )
# Add include directories
target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
GENERATE_EXPORT_HEADER( ANN
    BASE_NAME ANN
    EXPORT_MACRO_NAME ANN_EXPORT#Macro keyword for exported functions
    EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/include/${target}/ann_export.h#Auto generated header for dll
    STATIC_DEFINE ANN_BUILT_AS_STATIC#flag for static build
)
