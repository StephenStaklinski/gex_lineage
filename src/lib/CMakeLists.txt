# src/lib/CMakeLists.txt

# List your library sources here (mirror what your Makefile uses)
add_library(vine_core STATIC
    adam_scheduler.c
    backprop.c
    bitset.c
    brownian.c
    covariance.c
    crispr.c
    geometry.c
    gex.c
    gradients.c
    heap.c
    hutchinson.c
    likelihoods.c
    migration.c
    mcmc.c
    multi_mvn.c
    multiDAG.c
    mvn.c
    nj.c
    nuisance.c
    pca.c
    planar_flow.c
    radial_flow.c
    rf.c
    sparse_matrix.c
    sparse_vector.c
    taylor.c
    tree_prior.c
    upgma.c
    variational.c
)

find_package(OpenMP)

set_target_properties(vine_core PROPERTIES
    OUTPUT_NAME vine
)

target_compile_options(vine_core PUBLIC
    -g -fno-inline -Wall -fno-strict-aliasing
)

target_include_directories(vine_core
    PUBLIC
        "${VINE_INCLUDE_DIR}"     
)

target_link_libraries(vine_core
    PUBLIC
        phast::phast_core
)

if(OpenMP_C_FOUND)
  target_link_libraries(vine_core PUBLIC OpenMP::OpenMP_C)
  target_compile_definitions(vine_core PUBLIC VINE_HAS_OPENMP)
endif()

# Install the library
install(TARGETS vine_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
