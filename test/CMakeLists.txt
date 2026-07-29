include(CheckLanguage)
include(cmake/addFfxTest.cmake)

find_package(GTest REQUIRED)
find_package(OpenMP)
find_package(TBB)

if(NOT TARGET build_tests)
  add_custom_target(build_tests)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(FFX_NN_ENABLED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(GLOBAL_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
  file(MAKE_DIRECTORY "${GLOBAL_BIN_DIR}")

  set(PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")
  set(PYTHON_TEST_RUNNER "${PYTHON_DIR}/test_runner.py")

  file(GLOB NN_SOURCE_FILES CONFIGURE_DEPENDS "*_nn_*.cc")
  foreach(SOURCE_FILE ${NN_SOURCE_FILES})
    get_filename_component(TEST_NAME ${SOURCE_FILE} NAME_WE)

    set(PYTHON_TEST "${PYTHON_DIR}/${TEST_NAME}.py")
    set(BIN_FILENAME "${TEST_NAME}.bin")
    set(EXPECTED_BIN_FILE "${GLOBAL_BIN_DIR}/${BIN_FILENAME}")
    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/${BIN_FILENAME}.o")
    set(OBJ_LIB_NAME "${TEST_NAME}_bin_obj")

    if(EXISTS ${PYTHON_TEST})
      add_custom_command(
        OUTPUT "${EXPECTED_BIN_FILE}"
        COMMAND ${Python3_EXECUTABLE} "${PYTHON_TEST_RUNNER}" --test
                "${PYTHON_TEST}" --output "${GLOBAL_BIN_DIR}"
        DEPENDS "${PYTHON_TEST}" "${PYTHON_TEST_RUNNER}"
        COMMENT "Generating test blob ${BIN_FILENAME}"
        VERBATIM)

      add_custom_command(
        OUTPUT "${OBJ_FILE}"
        COMMAND ld -r -b binary -z noexecstack "${BIN_FILENAME}" -o
                "${OBJ_FILE}"
        DEPENDS "${EXPECTED_BIN_FILE}"
        WORKING_DIRECTORY "${GLOBAL_BIN_DIR}"
        COMMENT
          "Baking single binary workspace blob ${BIN_FILENAME} into native object layout"
        VERBATIM)

      add_library(${OBJ_LIB_NAME} STATIC "${OBJ_FILE}")
      set_target_properties(
        ${OBJ_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE
                                                       ON)
    endif()
  endforeach()
endif()

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS "*.cc")
if(NOT FFX_ALGORITHM_ENABLED)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_algorithm_.*\\.cc$")
endif()
if(NOT FFX_FRAMEWORK_ENABLED)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_framework_.*\\.cc$")
endif()
if(NOT FFX_NN_ENABLED)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_nn_.*\\.cc$")
endif()

add_subdirectory(serial)

if(OpenMP_CXX_FOUND)
  add_subdirectory(omp2)
endif()

if(TBB_FOUND)
  add_subdirectory(tbb)
endif()

if(NOT CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()

if(NOT CMAKE_HIP_COMPILER AND EXISTS "/opt/rocm/llvm/bin/clang++")
  set(CMAKE_HIP_COMPILER "/opt/rocm/llvm/bin/clang++")
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
  add_subdirectory(hip)
endif()
