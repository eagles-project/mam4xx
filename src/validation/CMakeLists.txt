# Here's a submodule that contains Skywalker inputs and Python scripts for
# processing outputs.
set(MAM_X_VALIDATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mam_x_validation)

# This library has some common utilities used by Skywalker drivers in this
# directory tree, including the functions in testing.cpp for creating column
# views, atmospheric state, prognostics, and diagnostics used by unit tests.
add_library(validation validation.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../tests/testing.cpp)
target_link_libraries(validation skywalker)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR}/include) # for skywalker

#--------------------------
# MAM4xx Skywalker drivers
#--------------------------
add_subdirectory(nucleation)
add_subdirectory(gasaerexch)
add_subdirectory(calcsize)
add_subdirectory(convproc)
add_subdirectory(coagulation)
add_subdirectory(aging)
add_subdirectory(rename)
add_subdirectory(nucleate_ice)
add_subdirectory(hetfrz)
add_subdirectory(aero_model)
add_subdirectory(wetdep)
add_subdirectory(drydep)
add_subdirectory(gas_chem)
add_subdirectory(ndrop)
add_subdirectory(water_uptake)
add_subdirectory(mo_photo)
add_subdirectory(lin_strat_chem)
add_subdirectory(mo_chm_diags)
add_subdirectory(mo_setext)
add_subdirectory(mo_setsox)
