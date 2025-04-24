if (ENABLE_SKYWALKER)
  include(ExternalProject)
  include(GNUInstallDirs)
  include_directories(${PROJECT_BINARY_DIR}/include)
  # required for the external project steps used below that prevent a
  # build/install race condition
  cmake_policy(SET CMP0114 NEW)

  # Create canonical top-level directories in the build directory.
  foreach(dir include;${CMAKE_INSTALL_LIBDIR};bin;share)
    set(top_level_dir ${PROJECT_BINARY_DIR}/${dir})
    if (NOT EXISTS ${top_level_dir})
      execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${top_level_dir})
    endif()
  endforeach()

  #----------------------------------------------------------------------------
  # Skywalker parameter study library for cross validation with legacy MAM4
  #----------------------------------------------------------------------------
  add_library(skywalker STATIC IMPORTED GLOBAL)
  set(SKYWALKER_LIBRARY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/libskywalker_${HAERO_PRECISION}.a)
  set_target_properties(skywalker PROPERTIES IMPORTED_LOCATION ${SKYWALKER_LIBRARY})
  if (NOT EXISTS ${SKYWALKER_LIBRARY})
    set(SW_CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
                      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                      -DENABLE_FORTRAN=OFF
                      -DSKYWALKER_PRECISION=${HAERO_PRECISION})
    ExternalProject_Add(skywalker_proj
                        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/skywalker
                        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/skywalker
                        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/skywalker
                        INSTALL_DIR ${PROJECT_BINARY_DIR}
                        CMAKE_ARGS ${SW_CMAKE_OPTS}
                        LOG_CONFIGURE TRUE
                        LOG_BUILD TRUE
                        LOG_INSTALL TRUE
                        STEP_TARGETS configure build install)
    ExternalProject_Add_StepDependencies(skywalker_proj build skywalker_proj-configure)
    ExternalProject_Add_StepDependencies(skywalker_proj install skywalker_proj-build)
    add_dependencies(skywalker skywalker_proj-install)
  endif()
endif()
