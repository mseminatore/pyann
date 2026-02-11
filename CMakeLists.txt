# Build the libann shared library for pyann
#
# This CMakeLists.txt is designed to be used as a subdirectory build
# that produces a shared library (.dll, .so, .dylib) from the ann source.
#
# Usage:
#   mkdir build && cd build
#   cmake ..
#   cmake --build . --config Release
#
# The resulting library will be copied to pyann/lib/

cmake_minimum_required(VERSION 3.14)
project(pyann_native VERSION 0.1.0 LANGUAGES C)

# Options
option(USE_BLAS "Use OpenBLAS for acceleration" OFF)
option(USE_CBLAS "Use CBLAS for acceleration" OFF)

# Find the ann source (expects it as a submodule or sibling directory)
set(ANN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ann" CACHE PATH "Path to ann source directory")

if(NOT EXISTS "${ANN_SOURCE_DIR}/ann.c")
    message(FATAL_ERROR 
        "Could not find ann source at ${ANN_SOURCE_DIR}\n"
        "Please either:\n"
        "  1. Clone ann as a submodule: git submodule add https://github.com/mseminatore/ann.git ann\n"
        "  2. Set ANN_SOURCE_DIR to the correct path: cmake -DANN_SOURCE_DIR=/path/to/ann .."
    )
endif()

# Source files for the shared library
set(ANN_SOURCES
    ${ANN_SOURCE_DIR}/ann.c
    ${ANN_SOURCE_DIR}/tensor.c
    ${ANN_SOURCE_DIR}/json.c
)

# Create shared library
add_library(ann SHARED ${ANN_SOURCES})

# Set properties for shared library
set_target_properties(ann PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# Include directories
target_include_directories(ann PUBLIC ${ANN_SOURCE_DIR})

# Platform-specific settings
if(WIN32)
    # Windows: export all symbols
    set_target_properties(ann PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set(LIB_OUTPUT_NAME "ann")
elseif(APPLE)
    set(LIB_OUTPUT_NAME "libann")
    set_target_properties(ann PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
else()
    # Linux
    set(LIB_OUTPUT_NAME "libann")
    target_link_libraries(ann PRIVATE m)  # Math library
    set_target_properties(ann PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# BLAS support
if(USE_CBLAS)
    target_compile_definitions(ann PRIVATE USE_CBLAS=1 USE_BLAS=1)
    
    # Find CBLAS
    if(WIN32)
        set(CBLAS_ROOT "C:/opt/cblas" CACHE PATH "CBLAS installation directory")
    else()
        set(CBLAS_ROOT "/opt/cblas" CACHE PATH "CBLAS installation directory")
    endif()
    
    if(EXISTS "${CBLAS_ROOT}/include/cblas.h")
        target_include_directories(ann PRIVATE ${CBLAS_ROOT}/include)
        target_link_directories(ann PRIVATE ${CBLAS_ROOT}/lib)
        target_link_libraries(ann PRIVATE cblas)
    else()
        message(WARNING "CBLAS not found at ${CBLAS_ROOT}")
    endif()
elseif(USE_BLAS)
    target_compile_definitions(ann PRIVATE USE_BLAS=1)
    
    if(WIN32)
        message(WARNING "OpenBLAS not supported on Windows, use USE_CBLAS instead")
    else()
        set(OPENBLAS_ROOT "/opt/OpenBLAS" CACHE PATH "OpenBLAS installation directory")
        if(EXISTS "${OPENBLAS_ROOT}/include/cblas.h")
            target_include_directories(ann PRIVATE ${OPENBLAS_ROOT}/include)
            target_link_directories(ann PRIVATE ${OPENBLAS_ROOT}/lib)
            target_link_libraries(ann PRIVATE openblas)
        else()
            # Try system OpenBLAS
            find_package(BLAS)
            if(BLAS_FOUND)
                target_link_libraries(ann PRIVATE ${BLAS_LIBRARIES})
            endif()
        endif()
    endif()
endif()

# Hypertune library (optional)
if(EXISTS "${ANN_SOURCE_DIR}/ann_hypertune.c")
    target_sources(ann PRIVATE ${ANN_SOURCE_DIR}/ann_hypertune.c)
endif()

# Copy library to pyann package directory
set(PYANN_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pyann/lib")
file(MAKE_DIRECTORY ${PYANN_LIB_DIR})

add_custom_command(TARGET ann POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ann> ${PYANN_LIB_DIR}/
    COMMENT "Copying shared library to ${PYANN_LIB_DIR}"
)

# Installation
install(TARGETS ann
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)

# Print summary
message(STATUS "")
message(STATUS "pyann native library configuration:")
message(STATUS "  ANN source: ${ANN_SOURCE_DIR}")
message(STATUS "  BLAS: ${USE_BLAS}")
message(STATUS "  CBLAS: ${USE_CBLAS}")
message(STATUS "  Output: ${PYANN_LIB_DIR}")
message(STATUS "")
