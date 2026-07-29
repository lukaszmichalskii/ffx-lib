set(BIN_DEPS_DIR "${GLOBAL_BIN_DIR}")

function(addFfxTest SOURCE_FILE BACKEND)
  get_filename_component(TEST_NAME ${SOURCE_FILE} NAME_WE)
  string(TOLOWER "${BACKEND}" BACKEND_LOWER)
  set(TARGET_NAME "${TEST_NAME}_${BACKEND_LOWER}")

  add_executable(${TARGET_NAME} ${SOURCE_FILE})

  target_link_libraries(${TARGET_NAME} PRIVATE ffx::ffx GTest::gtest_main)

  if(BACKEND_LOWER STREQUAL "serial")
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
              ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=64)
  elseif(BACKEND_LOWER STREQUAL "tbb")
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
              ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=64)
    target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
  elseif(BACKEND_LOWER STREQUAL "omp2")
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
              ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=64)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
  elseif(BACKEND_LOWER STREQUAL "cuda")
    target_compile_definitions(${TARGET_NAME}
                               PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
    set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
    target_compile_options(
      ${TARGET_NAME} PRIVATE --expt-relaxed-constexpr --expt-extended-lambda
                             -Xcompiler -std=c++23)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                    ON CUDA_ARCHITECTURES 120)
  elseif(BACKEND_LOWER STREQUAL "hip")
    set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE HIP)
    target_compile_definitions(${TARGET_NAME}
                               PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED)
    set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES gfx1201)
  else()
    message(
      FATAL_ERROR "Unsupported backend '${BACKEND}' provided to addFfxTest.")
  endif()

  if(FFX_NN_ENABLED AND "${TEST_NAME}" MATCHES ".*_nn_.*")
    set(OBJ_LIB_NAME "${TEST_NAME}_bin_obj")
    if(TARGET ${OBJ_LIB_NAME})
      target_link_libraries(${TARGET_NAME} PRIVATE ${OBJ_LIB_NAME})
    endif()
  endif()

  set_target_properties(
    ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                              OUTPUT_NAME ${TARGET_NAME})

  add_dependencies(build_tests ${TARGET_NAME})

  set(TEST_LABELS "cpp" "${BACKEND_LOWER}")

  if("${TARGET_NAME}" MATCHES ".*_nn_.*")
    list(APPEND TEST_LABELS "nn")
  elseif("${TARGET_NAME}" MATCHES ".*_algorithm_.*")
    list(APPEND TEST_LABELS "algorithm")
  elseif("${TARGET_NAME}" MATCHES ".*_framework_.*")
    list(APPEND TEST_LABELS "framework")
  endif()

  gtest_discover_tests(${TARGET_NAME} PROPERTIES LABELS "${TEST_LABELS}")
endfunction()
