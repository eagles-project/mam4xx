# Here's a submodule that contains Skywalker inputs and Python scripts for
# processing outputs.
set(MAM_X_VALIDATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mam_x_validation)

# This library has some common utilities used by Skywalker drivers in this
# directory tree.
add_library(validation validation.cpp)
target_link_libraries(validation skywalker)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#-------------------------
# MAM4xx Skywalker drivers
#-------------------------
add_subdirectory(nucleation)
add_subdirectory(gasaerexch)
add_subdirectory(calcsize)
add_subdirectory(coagulation)    
add_subdirectory(aging)
add_subdirectory(rename)
