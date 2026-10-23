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

cmake_minimum_required(VERSION 3.15)
project(pyann_native VERSION 0.1.0 LANGUAGES C)

# Options
option(USE_BLAS "Use OpenBLAS for acceleration" OFF)
option(USE_CBLAS "Use CBLAS for acceleration" OFF)
option(USE_ACCELERATE "Use Apple Accelerate for BLAS (macOS only)" OFF)
option(USE_SCIPY_OPENBLAS "Use the prebuilt OpenBLAS from the scipy-openblas32 Python package" OFF)
option(USE_METAL "Use Apple Metal GPU acceleration (macOS only)" OFF)
option(USE_CUDA "Use NVIDIA CUDA GPU acceleration" OFF)
set(USE_VULKAN OFF CACHE STRING "Vulkan GPU acceleration (AMD, Intel, NVIDIA): ON, OFF, or AUTO (on when it can be built)")
set(PYANN_VULKAN_PREBUILT "${CMAKE_CURRENT_SOURCE_DIR}/vulkan_prebuilt" CACHE PATH
    "Prebuilt Vulkan headers and shaders, used when glslc is not available (see tools/prepare_vulkan.py)")

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
    ${ANN_SOURCE_DIR}/ann_quantize.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/native/pyann_info.c
)

# Create shared library
add_library(ann SHARED ${ANN_SOURCES})

# libann's CPU worker threads (pthreads; nothing extra on Windows)
find_package(Threads REQUIRED)
target_link_libraries(ann PRIVATE Threads::Threads)

# Set properties for shared library
set_target_properties(ann PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    # Export only libann's ANN_API functions (and pyann_info.c's PYANN_EXPORT)
    C_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
)
# ANN_API becomes __declspec(dllexport) on Windows
target_compile_definitions(ann PRIVATE ANN_BUILDING_DLL)
# Wheels can't hold the symlinks a versioned library needs, so only
# versioned in development builds
if(NOT SKBUILD)
    set_target_properties(ann PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)
endif()

# Include directories
target_include_directories(ann PUBLIC ${ANN_SOURCE_DIR})

# Platform-specific settings
if(WIN32)
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
    set_target_properties(ann PROPERTIES C_STANDARD 11)

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
elseif(USE_ACCELERATE)
    if(NOT APPLE)
        message(FATAL_ERROR "USE_ACCELERATE is only supported on macOS")
    endif()
    # libann includes <cblas.h>; point it at Accelerate's copy.
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE MACOS_SDK
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(VECLIB_HEADERS "${MACOS_SDK}/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers")
    if(NOT EXISTS "${VECLIB_HEADERS}/cblas.h")
        message(FATAL_ERROR "Accelerate cblas.h not found at ${VECLIB_HEADERS}")
    endif()
    # Accelerate names cblas_saxpby catlas_saxpby (same signature)
    target_compile_definitions(ann PRIVATE USE_BLAS=1 cblas_saxpby=catlas_saxpby)
    target_include_directories(ann PRIVATE ${VECLIB_HEADERS})
    # The legacy CBLAS header is marked deprecated on macOS 13.3+
    target_compile_options(ann PRIVATE -Wno-deprecated-declarations)
    target_link_libraries(ann PRIVATE "-framework Accelerate")
elseif(USE_SCIPY_OPENBLAS)
    # Prebuilt, runtime-dispatched OpenBLAS (the one numpy and scipy ship) for
    # Linux and Windows wheels. Its symbols are prefixed with scipy_ so it can't
    # clash with another OpenBLAS in the same process.
    find_package(Python COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c
            "import scipy_openblas32 as s; from importlib.metadata import files; print(s.get_include_dir()); print(s.get_lib_dir()); print(next((f.locate().as_posix() for f in files('scipy-openblas32') if f.name == 'LICENSE.txt'), ''))"
        OUTPUT_VARIABLE SOB_PATHS
        RESULT_VARIABLE SOB_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT SOB_RESULT EQUAL 0)
        message(FATAL_ERROR "USE_SCIPY_OPENBLAS requires the scipy-openblas32 package (pip install scipy-openblas32)")
    endif()
    string(REGEX REPLACE "\r?\n" ";" SOB_PATHS "${SOB_PATHS}")
    list(GET SOB_PATHS 0 SOB_INCLUDE_DIR)
    list(GET SOB_PATHS 1 SOB_LIB_DIR)
    list(LENGTH SOB_PATHS SOB_PATH_COUNT)
    if(SOB_PATH_COUNT GREATER 2)
        list(GET SOB_PATHS 2 SOB_LICENSE)
    endif()
    find_library(SOB_LIBRARY NAMES scipy_openblas libscipy_openblas
                 PATHS ${SOB_LIB_DIR} NO_DEFAULT_PATH)
    if(NOT SOB_LIBRARY)
        message(FATAL_ERROR "scipy-openblas32 library not found in ${SOB_LIB_DIR}")
    endif()
    # libann calls the unprefixed names
    foreach(fn sgemm sgemv sger saxpy saxpby sscal sasum)
        target_compile_definitions(ann PRIVATE cblas_${fn}=scipy_cblas_${fn})
    endforeach()
    target_compile_definitions(ann PRIVATE USE_BLAS=1)
    target_include_directories(ann PRIVATE ${SOB_INCLUDE_DIR})
    target_link_libraries(ann PRIVATE ${SOB_LIBRARY})
    # Development builds load OpenBLAS from the package; wheels bundle a copy
    # next to libann instead (see the install rules below)
    if(NOT SKBUILD)
        set_property(TARGET ann APPEND PROPERTY INSTALL_RPATH "${SOB_LIB_DIR}")
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(ann PRIVATE _GNU_SOURCE)
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

# Report the BLAS in use through pyann_blas_backend()
if(USE_CBLAS)
    set(PYANN_BLAS_NAME "cblas")
elseif(USE_ACCELERATE)
    set(PYANN_BLAS_NAME "accelerate")
elseif(USE_BLAS OR USE_SCIPY_OPENBLAS)
    set(PYANN_BLAS_NAME "openblas")
endif()
if(PYANN_BLAS_NAME)
    set_source_files_properties(src/native/pyann_info.c PROPERTIES
        COMPILE_DEFINITIONS "PYANN_BLAS_NAME=\"${PYANN_BLAS_NAME}\"")
endif()

# Hypertune library (optional)
if(EXISTS "${ANN_SOURCE_DIR}/ann_hypertune.c")
    target_sources(ann PRIVATE ${ANN_SOURCE_DIR}/ann_hypertune.c)
endif()

# Metal GPU support (macOS only)
if(USE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "USE_METAL is only supported on macOS/Apple platforms")
    endif()
    target_compile_definitions(ann PRIVATE USE_METAL=1)
    target_sources(ann PRIVATE ${ANN_SOURCE_DIR}/tensor_metal.m)
    set_source_files_properties(${ANN_SOURCE_DIR}/tensor_metal.m PROPERTIES
        COMPILE_FLAGS "-x objective-c -fobjc-arc"
        LANGUAGE C
    )
    target_link_libraries(ann PRIVATE "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")
endif()

# CUDA GPU support
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(ann PRIVATE USE_CUDA=1)
    target_sources(ann PRIVATE ${ANN_SOURCE_DIR}/tensor_cuda.cu ${ANN_SOURCE_DIR}/shaders.cu)
    target_link_libraries(ann PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Vulkan GPU support: ON, OFF, or AUTO (on when it can be built, else skipped).
# libann opens the Vulkan loader at run time, so the library has no link
# dependency on it and falls back to the CPU on machines without a Vulkan
# driver. Building needs the Vulkan headers and the compiled shaders: with the
# Vulkan SDK (or libvulkan-dev and glslc) installed, the shaders are compiled
# from ann/; otherwise they come from vulkan_prebuilt/, which
# tools/prepare_vulkan.py writes and the sdist ships.
string(TOUPPER "${USE_VULKAN}" VULKAN_MODE)
set(VULKAN_BUILT OFF)
if(VULKAN_MODE STREQUAL "AUTO" OR USE_VULKAN)
    find_package(Vulkan QUIET)
    if(Vulkan_INCLUDE_DIRS AND Vulkan_GLSLC_EXECUTABLE)
        # Compile each shader to SPIR-V, then embed them all in shaders_vk_spv.h
        # (the same steps as libann's own CMakeLists.txt)
        set(VULKAN_SHADERS gemm bias_act softmax act_deriv softmax_jacobian output_delta_loss
                           bias_grad_sum optimizer dropout
                           conv_im2col conv_output conv_grad_rows conv_col2im
                           maxpool_forward maxpool_backward upsample sum_partials
                           conv_direct_forward conv_direct_input_grad conv_direct_weight_grad)
        set(VULKAN_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/vk_spv)
        file(MAKE_DIRECTORY ${VULKAN_SPV_DIR})
        set(VULKAN_SPV_FILES "")
        foreach(shader ${VULKAN_SHADERS})
            add_custom_command(
                OUTPUT ${VULKAN_SPV_DIR}/${shader}.spv
                COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.1 -O
                        -I ${ANN_SOURCE_DIR}/shaders/vk
                        -o ${VULKAN_SPV_DIR}/${shader}.spv ${ANN_SOURCE_DIR}/shaders/vk/${shader}.comp
                DEPENDS ${ANN_SOURCE_DIR}/shaders/vk/${shader}.comp ${ANN_SOURCE_DIR}/shaders/vk/common.glsl
                COMMENT "Compiling Vulkan shader ${shader}.comp"
                VERBATIM)
            list(APPEND VULKAN_SPV_FILES ${VULKAN_SPV_DIR}/${shader}.spv)
        endforeach()
        # comma-separated: a ';' list would end the command in a Unix shell
        string(REPLACE ";" "," VULKAN_SHADER_NAMES "${VULKAN_SHADERS}")
        add_custom_command(
            OUTPUT ${VULKAN_SPV_DIR}/shaders_vk_spv.h
            COMMAND ${CMAKE_COMMAND} -DSPV_DIR=${VULKAN_SPV_DIR} -DNAMES=${VULKAN_SHADER_NAMES}
                    -DOUT=${VULKAN_SPV_DIR}/shaders_vk_spv.h -P ${ANN_SOURCE_DIR}/cmake/embed_spirv.cmake
            DEPENDS ${VULKAN_SPV_FILES} ${ANN_SOURCE_DIR}/cmake/embed_spirv.cmake
            COMMENT "Embedding Vulkan shaders"
            VERBATIM)
        # tools/prepare_vulkan.py builds just this
        add_custom_target(pyann_vk_shaders DEPENDS ${VULKAN_SPV_DIR}/shaders_vk_spv.h)
        set(VULKAN_INCLUDES ${Vulkan_INCLUDE_DIRS} ${VULKAN_SPV_DIR})
        set(VULKAN_SOURCE "shaders compiled with ${Vulkan_GLSLC_EXECUTABLE}")
        set(VULKAN_BUILT ON)
    elseif(EXISTS "${PYANN_VULKAN_PREBUILT}/shaders_vk_spv.h" AND
           EXISTS "${PYANN_VULKAN_PREBUILT}/include/vulkan/vulkan.h")
        set(VULKAN_SPV_DIR ${PYANN_VULKAN_PREBUILT})
        set(VULKAN_INCLUDES ${PYANN_VULKAN_PREBUILT}/include ${PYANN_VULKAN_PREBUILT})
        set(VULKAN_SOURCE "prebuilt shaders from ${PYANN_VULKAN_PREBUILT}")
        set(VULKAN_BUILT ON)
    elseif(VULKAN_MODE STREQUAL "AUTO")
        message(STATUS "Vulkan headers and glslc not found, and no ${PYANN_VULKAN_PREBUILT}: building without Vulkan")
    else()
        message(FATAL_ERROR "USE_VULKAN needs the Vulkan headers and glslc (install the Vulkan SDK, "
                            "or libvulkan-dev and glslc), or the shaders prebuilt by tools/prepare_vulkan.py")
    endif()
endif()
if(VULKAN_BUILT)
    target_compile_definitions(ann PRIVATE USE_VULKAN=1)
    target_sources(ann PRIVATE ${ANN_SOURCE_DIR}/tensor_vulkan.c ${VULKAN_SPV_DIR}/shaders_vk_spv.h)
    target_include_directories(ann PRIVATE ${VULKAN_INCLUDES})
    # dlopen() on Linux
    target_link_libraries(ann PRIVATE ${CMAKE_DL_LIBS})
else()
    set(VULKAN_SOURCE "off")
endif()

if(SKBUILD)
    # Building a wheel: install the library into the pyann package
    install(TARGETS ann
        LIBRARY DESTINATION pyann/lib
        RUNTIME DESTINATION pyann/lib
    )
    # The runtime exported C headers need (see Sequential.export_c_header);
    # development builds use the copy in the ann/ submodule
    install(FILES ${ANN_SOURCE_DIR}/ann_rt.h DESTINATION pyann/include)
    if(USE_SCIPY_OPENBLAS)
        # Ship OpenBLAS (and its runtime dependencies) next to libann so the
        # wheel works without scipy-openblas32 installed, including source
        # installs that never go through auditwheel or delvewheel.
        file(GLOB SOB_RUNTIME_LIBS "${SOB_LIB_DIR}/*.so*" "${SOB_LIB_DIR}/*.dll" "${SOB_LIB_DIR}/*.dylib")
        install(FILES ${SOB_RUNTIME_LIBS} DESTINATION pyann/lib)
        # OpenBLAS, LAPACK, libgfortran and libquadmath carry their own licenses
        if(SOB_LICENSE AND EXISTS "${SOB_LICENSE}")
            install(FILES "${SOB_LICENSE}" DESTINATION pyann/lib RENAME LICENSE-scipy-openblas32.txt)
        else()
            message(FATAL_ERROR "scipy-openblas32 license file not found; it must ship with the bundled libraries")
        endif()
    endif()
    set(PYANN_LIB_DIR "wheel: pyann/lib")
else()
    # Development build: copy the library next to the sources
    set(PYANN_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/pyann/lib")
    file(MAKE_DIRECTORY ${PYANN_LIB_DIR})

    add_custom_command(TARGET ann POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ann> ${PYANN_LIB_DIR}/
        COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:ann> ${PYANN_LIB_DIR}/libann${CMAKE_SHARED_LIBRARY_SUFFIX}
        COMMENT "Copying shared library to ${PYANN_LIB_DIR}"
    )

    install(TARGETS ann
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
    )
endif()

# Print summary
message(STATUS "")
message(STATUS "pyann native library configuration:")
message(STATUS "  ANN source: ${ANN_SOURCE_DIR}")
message(STATUS "  BLAS: ${USE_BLAS}")
message(STATUS "  CBLAS: ${USE_CBLAS}")
message(STATUS "  Accelerate: ${USE_ACCELERATE}")
message(STATUS "  scipy-openblas32: ${USE_SCIPY_OPENBLAS}")
message(STATUS "  Metal: ${USE_METAL}")
message(STATUS "  CUDA: ${USE_CUDA}")
message(STATUS "  Vulkan: ${USE_VULKAN} (${VULKAN_SOURCE})")
message(STATUS "  Output: ${PYANN_LIB_DIR}")
message(STATUS "")
