# One executable per program.

add_executable(gexFilter gexFilter.c)
add_executable(gexPca gexPca.c)
add_executable(gexFactor gexFactor.c)
add_executable(gexSim gexSim.c)
add_executable(gexEvalSim gexEvalSim.c)

# ---- Optional codesigning (macOS) ----
# Default is ON on macOS for normal developer builds, but OFF under conda-build.
set(_gex_lineage_codesign_default OFF)
if(APPLE)
  set(_gex_lineage_codesign_default ON)
  if(DEFINED ENV{CONDA_BUILD})
    set(_gex_lineage_codesign_default OFF)
  endif()
endif()

option(GEX_LINEAGE_CODESIGN "Ad-hoc codesign executables on macOS after build" ${_gex_lineage_codesign_default})
message(STATUS "GEX_LINEAGE_CODESIGN=${GEX_LINEAGE_CODESIGN} (CONDA_BUILD=$ENV{CONDA_BUILD})")

foreach(tgt gexFilter gexPca gexFactor gexSim gexEvalSim)
    target_link_libraries(${tgt} PRIVATE gex_lineage_core)
    if(APPLE AND GEX_LINEAGE_CODESIGN)
        add_custom_command(TARGET ${tgt} POST_BUILD
            COMMAND /usr/bin/codesign --force --sign - "$<TARGET_FILE:${tgt}>"
            COMMENT "Ad-hoc code signing ${tgt}"
            VERBATIM
        )
    endif()
endforeach()

# Read version file and strip whitespace
file(READ "${CMAKE_SOURCE_DIR}/version" GEX_LINEAGE_VERSION_RAW)
string(STRIP "${GEX_LINEAGE_VERSION_RAW}" GEX_LINEAGE_VERSION)
message(STATUS "GEX_LINEAGE_VERSION='${GEX_LINEAGE_VERSION}'")

foreach(tgt gexFilter gexPca gexFactor gexSim gexEvalSim)
    target_compile_definitions(${tgt} PRIVATE GEX_LINEAGE_VERSION="${GEX_LINEAGE_VERSION}")
endforeach()


# Install all progs into bin/
install(TARGETS
  gexFilter
  gexPca
  gexFactor
  gexSim
  gexEvalSim
  RUNTIME DESTINATION bin
)
