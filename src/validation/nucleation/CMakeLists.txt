set(NUC_VALIDATION_DIR ${MAM_X_VALIDATION_DIR}/nucleation)
set(NUC_BASELINE_DIR ${MAM4BOX_BASELINE_DIR}/nucleation)

# These subdirectories contain Skywalker drivers for MAM4 parameterizations.
# Include directory for .mod files.
include_directories(${PROJECT_BINARY_DIR}/haero)
include_directories(${PROJECT_BINARY_DIR}/validation)

# We use a single driver for all nucleation-related parameterizations.
add_executable(nucleation_driver nucleation_driver.cpp
               mer07_veh02_wang08_nuc_1box.cpp
               newnuc_cluster_growth.cpp)
target_link_libraries(nucleation_driver skywalker;validation;${HAERO_LIBRARIES})

# Run the driver in several configurations to produce datasets.

# Nucleation rate
foreach (input
         vehkamaki2002_fig8
         vehkamaki2002_fig9
         vehkamaki2002_fig10
         vehkamaki2002_fig11
         vehkamaki2002_contour
         merikanto2007_fig2
         merikanto2007_fig3
         merikanto2007_fig4)
  add_test(validate_${input} nucleation_driver ${NUC_VALIDATION_DIR}/${input}.yaml)
endforeach()

# Nucleation particle cluster growth
foreach (input
         nucgrowth_case1
         nucgrowth_case2
         nucgrowth_case3)
  add_test(validate_${input} nucleation_driver ${NUC_VALIDATION_DIR}/${input}.yaml)
endforeach()

# Copy generate_plots.py to our binary directory for ease of use.
configure_file(
  ${NUC_VALIDATION_DIR}/generate_plots.py
  ${CMAKE_CURRENT_BINARY_DIR}/generate_plots.py
  COPYONLY
)
