include(CheckLanguage)
include(ffxAddTest)

find_package(GTest REQUIRED)
find_package(OpenMP)
find_package(TBB)

if(NOT TARGET build_tests)
  add_custom_target(build_tests)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(
  GLOB
  SOURCE_FILES
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

if(NOT FFX_ALGORITHM_ENABLED)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_algorithm_.*\\.cc$")
endif()
if(NOT FFX_FRAMEWORK_ENABLED)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_framework_.*\\.cc$")
endif()
if(NOT FFX_NN_ENABLED)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_nn_.*\\.cc$")
endif()
if(NOT FFX_SOA_ENABLED)
  list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_soa_.*\\.cc$")
endif()

set(HOST_ONLY_SOURCES "")
set(CUDA_SOURCES "")
set(HIP_SOURCES "")
set(PORTABLE_SOURCES "")

foreach(src ${SOURCE_FILES})
  if(src MATCHES "\\.dev\\.cc$")
    list(APPEND PORTABLE_SOURCES ${src})
  elseif(src MATCHES "\\.cu$")
    list(APPEND CUDA_SOURCES ${src})
  elseif(src MATCHES "\\.hip\\.cc$")
    list(APPEND HIP_SOURCES ${src})
  else()
    list(APPEND HOST_ONLY_SOURCES ${src})
  endif()
endforeach()

if(FFX_NN_ENABLED)
  set(ACTIVE_NN_SOURCES "")
  foreach(src ${PORTABLE_SOURCES})
    if(src MATCHES ".*_nn_.*")
      list(APPEND ACTIVE_NN_SOURCES ${src})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES ACTIVE_NN_SOURCES)

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(GLOBAL_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
  file(MAKE_DIRECTORY "${GLOBAL_BIN_DIR}")

  set(PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")
  set(PYTHON_TEST_RUNNER "${PYTHON_DIR}/test_runner.py")

  foreach(SOURCE_FILE ${ACTIVE_NN_SOURCES})
    get_filename_component(FILE_NAME ${SOURCE_FILE} NAME)
    string(REGEX REPLACE "(_nn_[a-zA-Z0-9_]+)\\..*" "\\1" TEST_NAME
                         ${FILE_NAME})

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
        COMMENT "Baking binary blob ${BIN_FILENAME} into native object layout"
        VERBATIM)

      if(NOT TARGET ${OBJ_LIB_NAME})
        add_library(${OBJ_LIB_NAME} STATIC "${OBJ_FILE}")
        set_target_properties(
          ${OBJ_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX
                                     POSITION_INDEPENDENT_CODE ON)
      endif()
    endif()
  endforeach()
endif()

# cpu host only sources
foreach(HOST_ONLY_SOURCE ${HOST_ONLY_SOURCES})
  get_filename_component(TEST_NAME ${HOST_ONLY_SOURCE} NAME_WE)
  add_executable(${TEST_NAME} ${HOST_ONLY_SOURCE})
  target_link_libraries(${TEST_NAME} PRIVATE ffx::ffx GTest::gtest_main)
  target_compile_definitions(${TEST_NAME}
                             PRIVATE ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128)
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

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
