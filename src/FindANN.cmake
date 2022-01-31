#   Note: only tested to work in the current setup, not general purpose.
#   Find ANN headers and libraries using input ANN_ROOT as the root
#
#   Try to find ANN: Approximate Nearest Neighbor Library
#   This module defines
#   ANN_INCLUDE_DIRS
#   ANN_LIBRARIES
#   ANN_FOUND
#
#   The following variable can be set as an argument for the module:
#   ANN_ROOT
include(FindPackageHandleStandardArgs)
#Find include 
find_path(
    ANN_INCLUDE_DIR
    NAMES ANN/ANN.h
    PATHS
    ${ANN_ROOT}/include
    DOC "Directory that contains ANN/ANN.h"
)
#Find compiled library
find_library( ANN_ann_LIBRARY
    NAMES
        ANN
    HINTS
        "${ANN_ROOT}"
        "${ANN_ROOT}/lib"
        "$ENV{ANN_ROOT}/lib"
    DOC "The Approximate Nearest Neighbor Library"
)
find_package_handle_standard_args(ANN DEFAULT_MSG ANN_INCLUDE_DIR)

if (ANN_FOUND)
    set(ANN_INCLUDE_DIRS ${ANN_INCLUDE_DIR})
    set(ANN_LIBRARIES ${ANN_ann_LIBRARY})
    set(GLFW_LIBRARY ${ANN_ann_LIBRARY})
endif()