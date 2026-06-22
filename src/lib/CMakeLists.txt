# src/lib/CMakeLists.txt

add_library(gex_lineage_core STATIC
    adam.c
    brownian.c
    latentflow.c
    gexmatrix.c
    misc.c
    model.c
    parser.c
    pca.c
    phylofilter.c
    mvn.c
)

find_package(OpenMP)

target_compile_options(gex_lineage_core PUBLIC
    -Wall 
    -fno-strict-aliasing
    $<$<CONFIG:Debug>:-g -fno-inline>
)

target_include_directories(gex_lineage_core
    PUBLIC
        "${GEX_LINEAGE_INCLUDE_DIR}"
)

target_link_libraries(gex_lineage_core
    PUBLIC
        phast::phast_core
)

if(OpenMP_C_FOUND)
  target_link_libraries(gex_lineage_core PUBLIC OpenMP::OpenMP_C)
  target_compile_definitions(gex_lineage_core PUBLIC GEX_LINEAGE_HAS_OPENMP)
endif()

if(GEX_LINEAGE_HAS_OPENBLAS_THREAD_CONTROL)
  target_compile_definitions(gex_lineage_core PUBLIC GEX_LINEAGE_HAS_OPENBLAS_THREAD_CONTROL)
endif()
if(GEX_LINEAGE_HAS_MKL_THREAD_CONTROL)
  target_compile_definitions(gex_lineage_core PUBLIC GEX_LINEAGE_HAS_MKL_THREAD_CONTROL)
endif()

# Install the library
install(TARGETS gex_lineage_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
