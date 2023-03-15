cmake_minimum_required (VERSION 3.17.0)

message(STATUS "Configuring with build type: ${CMAKE_BUILD_TYPE}")

# We rely on Haero for basic data structures and third-party libraries.
if (NOT MAM4XX_HAERO_DIR)
  message(FATAL_ERROR "MAM4XX_HAERO_DIR is not specified! Please provide a path to Haero.")
elseif (NOT EXISTS ${MAM4XX_HAERO_DIR}/share)
  message(FATAL_ERROR "Invalid MAM4XX_HAERO_DIR. Please provide a valid path to Haero.")
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MAM4XX_HAERO_DIR}/share/")
include(haero)

if(HAERO_ENABLE_GPU)
  message(STATUS "Building for GPU")
else() #CPU
  message(STATUS "Building for CPU")
endif()

option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(ENABLE_SKYWALKER "Enable Skywalker cross validation" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Blessed version of clang-format.
set(CLANG_FORMAT_VERSION 14)

# Set compilers, linkers, and flags from Haero.
set(CMAKE_CXX_STANDARD ${HAERO_CXX_STANDARD})
set(CMAKE_CXX_COMPILER ${HAERO_CXX_COMPILER})
set(CMAKE_CXX_FLAGS    ${HAERO_CXX_FLAGS})

set(CMAKE_C_STANDARD ${HAERO_C_STANDARD})
set(CMAKE_C_COMPILER ${HAERO_C_COMPILER})
set(CMAKE_C_FLAGS    ${HAERO_C_FLAGS})

enable_language(CXX)

project (mam4xx)

set(CMAKE_EXE_LINKER_FLAGS ${HAERO_EXE_LINKER_FLAGS})

message(STATUS "Generating project files in build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "C++ compiler is ${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
message(STATUS "C compiler is ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")

# Version numbers.
set(MAM4XX_MAJOR_VERSION 0)
set(MAM4XX_MINOR_VERSION 1)
set(MAM4XX_PATCH_VERSION 0)

set(MAM4XX_VERSION "${MAM4XX_MAJOR_VERSION}.${MAM4XX_MINOR_VERSION}.${MAM4XX_PATCH_VERSION}")
message(STATUS "Configuring MAM4xx v${MAM4XX_VERSION}")

# Precision of floating point numbers.
message(STATUS "Using ${HAERO_PRECISION} precision floating point numbers")

# We build static libraries only.
set(BUILD_SHARED_LIBS OFF)

# Report the installation prefix.
message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

set(MAM4XX_LIBRARIES ${HAERO_LIBRARIES})

# We look for headers in these directories.
include_directories(${HAERO_INCLUDE_DIRS})
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${MAM4XX_HAERO_DIR}/include")
include_directories(${PROJECT_SOURCE_DIR}/src)

link_directories("${MAM4XX_HAERO_DIR}/lib")

# Code coverage
if (ENABLE_COVERAGE)
  message(STATUS "Enabling code coverage instrumentation")
  find_program(LCOV_EXE lcov DOC "Lcov code coverage tool")
  if (LCOV_EXE MATCHES "NOTFOUND")
    message(FATAL_ERROR "Could not find lcov for code coverage reporting!")
  endif()

  # Add code coverage compiler/linker flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

  # Add a "make coverage" target.
  add_custom_target(coverage ${LCOV_EXE} --capture --directory . -o unfiltered_coverage.info
    COMMAND ${LCOV_EXE} --remove unfiltered_coverage.info '*/external/*' -o coverage.info
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT "Generating coverage report (coverage.info)")
endif()

# Support for valgrind (Linux only)
include(add_memcheck_target)
add_memcheck_target()

# Testing
include(CTest)
enable_testing()

# Third-party libraries.
add_subdirectory(ext)

# Source directories.
add_subdirectory(src)

# Formatting and format checking using clang-format.
find_program(CLANG_FORMAT clang-format)
if (NOT CLANG_FORMAT STREQUAL "CLANG_FORMAT-NOTFOUND")
  # Is this the blessed version? If not, we create targets that warn the user
  # to obtain the right version.
  execute_process(COMMAND clang-format --version
    OUTPUT_VARIABLE CF_VERSION)
  string(STRIP ${CF_VERSION} CF_VERSION)
  if (NOT ${CF_VERSION} MATCHES ${CLANG_FORMAT_VERSION})
    add_custom_target(format-cxx
      echo "You have clang-format version ${CF_VERSION}, but ${CLANG_FORMAT_VERSION} is required."
      "Please make sure this version appears in your path and rerun config.sh.")
    add_custom_target(format-cxx-check
      echo "You have clang-format version ${CF_VERSION}, but ${CLANG_FORMAT_VERSION} is required."
      "Please make sure this version appears in your path and rerun config.sh.")
  else()
    add_custom_target(format-cxx
      find ${PROJECT_SOURCE_DIR}/src -name "*.[hc]pp" -exec ${CLANG_FORMAT} -i {} \+;
      VERBATIM
      COMMENT "Auto-formatting C++ code...")
    add_custom_target(format-cxx-check
      find ${PROJECT_SOURCE_DIR}/src -name "*.[hc]pp" -exec ${CLANG_FORMAT} -n --Werror -ferror-limit=1 {} \+;
      VERBATIM
      COMMENT "Checking C++ formatting...")
  endif()

endif()
