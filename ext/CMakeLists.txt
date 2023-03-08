if (ENABLE_SKYWALKER)
  include(ExternalProject)
  include(GNUInstallDirs)
  include_directories(${PROJECT_BINARY_DIR}/include)

  # Create canonical top-level directories in the build directory.
  foreach(dir include;lib;bin;share)
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
                        BUILD_COMMAND make -j
                        LOG_BUILD TRUE
                        INSTALL_COMMAND make install
                        LOG_INSTALL TRUE)
    add_dependencies(skywalker skywalker_proj)
  endif()
endif()
