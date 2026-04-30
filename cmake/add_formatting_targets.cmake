# Blessed version of clang-format.
set(CLANG_FORMAT_VERSION 14)

# This macro creates the following targets for checking code formatting:
# make format-c       <-- reformats C code to conform to desired style
# make format-c-check <-- checks C code formatting, reporting any errors
macro(add_formatting_targets)
  if (NOT TARGET format-cxx)
    find_program(CLANG_FORMAT NAMES clang-format-${CLANG_FORMAT_VERSION} clang-format)
    if (NOT CLANG_FORMAT STREQUAL "CLANG_FORMAT-NOTFOUND")
      # Is this our blessed version? If not, we create targets that warn the user
      # to obtain the right version.
      execute_process(COMMAND ${CLANG_FORMAT} --version
        OUTPUT_VARIABLE CF_VERSION)
      string(STRIP ${CF_VERSION} CF_VERSION)
      if (NOT ${CF_VERSION} MATCHES ${CLANG_FORMAT_VERSION})
        add_custom_target(format-cxx
          echo "You have clang-format version ${CF_VERSION}, but ${CLANG_FORMAT_VERSION} is required."
          "Please make sure this version appears in your path and rerun cmake.")
        add_custom_target(format-cxx-check
          echo "You have clang-format version ${CF_VERSION}, but ${CLANG_FORMAT_VERSION} is required."
          "Please make sure this version appears in your path and rerun cmake.")
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
  endif()
endmacro()
