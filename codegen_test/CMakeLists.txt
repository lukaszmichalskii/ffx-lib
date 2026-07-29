cmake_minimum_required(VERSION 3.25)
project(codegen_test CXX)

set(MODEL_DATA "${CMAKE_CURRENT_SOURCE_DIR}/model.data")
set(MAIN_CC "${CMAKE_CURRENT_SOURCE_DIR}/main.cc")

if(EXISTS "${MODEL_DATA}")
  set(MODEL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/model.data.o")

  add_custom_command(
    OUTPUT "${MODEL_OBJ}"
    COMMAND ld -r -b binary -z noexecstack "model.data" -o "${MODEL_OBJ}"
    DEPENDS "${MODEL_DATA}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Baking model.data into native object layout"
    VERBATIM)

  add_library(model_data STATIC "${MODEL_OBJ}")
  set_target_properties(model_data PROPERTIES LINKER_LANGUAGE CXX
                                              POSITION_INDEPENDENT_CODE ON)
endif()

add_subdirectory(serial)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_subdirectory(omp2)
endif()

find_package(TBB)
if(TBB_FOUND)
  add_subdirectory(tbb)
endif()

include(CheckLanguage)

check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
  add_subdirectory(hip)
endif()
