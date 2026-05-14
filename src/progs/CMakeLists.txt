# one executable per program, linked against vine + PHAST

add_executable(vine vine.c)
add_executable(evalTrees evalTrees.c)
add_executable(crisprLnl crisprLnl.c)
add_executable(gexLineage gexLineage.c)
add_executable(gexSim gexSim.c)
add_executable(gexEvalSim gexEvalSim.c)

# ---- Optional codesigning (macOS) ----
# Default is ON on macOS for normal developer builds, but OFF under conda-build.
set(_vine_codesign_default OFF)
if(APPLE)
  set(_vine_codesign_default ON)
  if(DEFINED ENV{CONDA_BUILD})
    set(_vine_codesign_default OFF)
  endif()
endif()

option(VINE_CODESIGN "Ad-hoc codesign executables on macOS after build" ${_vine_codesign_default})
message(STATUS "VINE_CODESIGN=${VINE_CODESIGN} (CONDA_BUILD=$ENV{CONDA_BUILD})")

foreach(tgt vine evalTrees crisprLnl gexLineage gexSim gexEvalSim)
    target_link_libraries(${tgt} PRIVATE vine_core)
    if(APPLE AND VINE_CODESIGN)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND /usr/bin/codesign --force --sign - "$<TARGET_FILE:${tgt}>"
            COMMENT "Ad-hoc code signing ${tgt}"
            VERBATIM
        )
    endif()
endforeach()

if(VINE_HAS_OPENBLAS_THREAD_CONTROL)
    target_compile_definitions(gexLineage PRIVATE VINE_HAS_OPENBLAS_THREAD_CONTROL)
endif()
if(VINE_HAS_MKL_THREAD_CONTROL)
    target_compile_definitions(gexLineage PRIVATE VINE_HAS_MKL_THREAD_CONTROL)
endif()

# Read version file and strip whitespace
file(READ "${CMAKE_SOURCE_DIR}/version" VINE_VERSION_RAW)
string(STRIP "${VINE_VERSION_RAW}" VINE_VERSION)
message(STATUS "VINE_VERSION='${VINE_VERSION}'")

# add to vine compile target
target_compile_definitions(vine PRIVATE VINE_VERSION="${VINE_VERSION}")


# Install all progs into bin/
install(TARGETS
  vine
  evalTrees
  crisprLnl
  gexLineage
  gexSim
  gexEvalSim
  RUNTIME DESTINATION bin
)
