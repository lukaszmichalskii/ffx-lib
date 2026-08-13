cmake_minimum_required(VERSION 3.25)
project(
  ffx_mnist
  VERSION 1.0.0
  LANGUAGES CXX)

include(ffxAddExecutable)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET ffx AND NOT TARGET ffx::ffx)
  find_package(ffx REQUIRED)
endif()

set(MAIN_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/main.cc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(MODEL_DATA "${CMAKE_CURRENT_SOURCE_DIR}/bin/model.data")
set(DATA_SRC "${CMAKE_CURRENT_SOURCE_DIR}/bin/data.raw")
set(DATA_DST "${CMAKE_BINARY_DIR}/bin/data.raw")

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/bin/data.raw"
     DESTINATION "${CMAKE_BINARY_DIR}/bin")

add_compile_definitions($<$<CONFIG:Debug>:DEBUG>)
# ------------------------------------------------------------------------------
# 1. Bake model.data into static library
# ------------------------------------------------------------------------------
if(EXISTS "${MODEL_DATA}")
  set(MODEL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/model.data.o")

  add_custom_command(
    OUTPUT "${MODEL_OBJ}"
    COMMAND ld -r -b binary -z noexecstack "model.data" -o "${MODEL_OBJ}"
    DEPENDS "${MODEL_DATA}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
    COMMENT "Baking model.data into native object layout"
    VERBATIM)

  add_library(model_data STATIC "${MODEL_OBJ}")
  set_target_properties(model_data PROPERTIES LINKER_LANGUAGE CXX
                                              POSITION_INDEPENDENT_CODE ON)
endif()

# ------------------------------------------------------------------------------
# 1. Subdirectories
# ------------------------------------------------------------------------------
add_subdirectory(cmake/serial)

include(CheckLanguage)

if(NOT CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cmake/cuda)
endif()

if(NOT CMAKE_HIP_COMPILER AND EXISTS "/opt/rocm/llvm/bin/clang++")
  set(CMAKE_HIP_COMPILER "/opt/rocm/llvm/bin/clang++")
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
  add_subdirectory(cmake/hip)
endif()
