cmake_minimum_required(VERSION 3.16)

project(vine
  VERSION 0.2.0
  LANGUAGES C)

include(CheckIncludeFile)

# Use C99 
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# allows use of upper case PHAST_ROOT
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# ---- PHAST dependency --------------------------------------------------
# Users/packagers provide -DPHAST_ROOT=/path/to/prefix (the install prefix)
set(PHAST_ROOT "" CACHE PATH "PHAST install prefix (contains lib/cmake/phast)")

if (NOT PHAST_ROOT)
  message(FATAL_ERROR "PHAST_ROOT is not set. Configure with -DPHAST_ROOT=/path/to/phast-prefix")
endif()

# Tell CMake where to find phastConfig.cmake
list(PREPEND CMAKE_PREFIX_PATH "${PHAST_ROOT}")

find_package(phast CONFIG REQUIRED)

check_include_file("openblas/cblas.h" VINE_HAS_OPENBLAS_CBLAS_H)
check_include_file("mkl.h" VINE_HAS_MKL_H)

set(VINE_HAS_OPENBLAS_THREAD_CONTROL OFF)
set(VINE_HAS_MKL_THREAD_CONTROL OFF)
if(TARGET phast::phast_linalg)
  get_target_property(_vine_phast_linalg_libs phast::phast_linalg INTERFACE_LINK_LIBRARIES)
  if(_vine_phast_linalg_libs MATCHES "[Oo]pen[Bb][Ll][Aa][Ss]" AND VINE_HAS_OPENBLAS_CBLAS_H)
    set(VINE_HAS_OPENBLAS_THREAD_CONTROL ON)
  endif()
  if(_vine_phast_linalg_libs MATCHES "[Mm][Kk][Ll]" AND VINE_HAS_MKL_H)
    set(VINE_HAS_MKL_THREAD_CONTROL ON)
  endif()
endif()

# ---- Project layout ----------------------------------------------------

# public headers 
set(VINE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

add_subdirectory(src/lib)
add_subdirectory(src/progs)

# ---- Installation (for Homebrew, etc.) ---------------------------------
# Install public headers
install(DIRECTORY "${VINE_INCLUDE_DIR}/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
