function(ffx_add_executable SOURCE_FILE BACKEND)
  get_filename_component(FILE_NAME ${SOURCE_FILE} NAME_WE)
  string(TOLOWER "${BACKEND}" BACKEND_LOWER)
  set(TARGET_NAME "${FILE_NAME}_${BACKEND_LOWER}")

  add_executable(${TARGET_NAME} ${SOURCE_FILE})

  target_link_libraries(${TARGET_NAME} PRIVATE ffx::ffx)

  if(TARGET model_data)
    target_link_libraries(${TARGET_NAME} PRIVATE model_data)
  endif()

  if(BACKEND_LOWER STREQUAL "serial")
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
              ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128)
  elseif(BACKEND_LOWER STREQUAL "tbb")
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
              ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128)
    target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
  elseif(BACKEND_LOWER STREQUAL "omp2")
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
              ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=128)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenMP::OpenMP_CXX)
  elseif(BACKEND_LOWER STREQUAL "cuda")
    target_compile_definitions(${TARGET_NAME}
                               PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
    set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
    target_compile_options(
      ${TARGET_NAME} PRIVATE --expt-relaxed-constexpr --expt-extended-lambda
                             -Xcompiler -std=c++23 --diag-suppress=20012)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                    ON CUDA_ARCHITECTURES 120)
  elseif(BACKEND_LOWER STREQUAL "hip")
    set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE HIP)
    target_compile_definitions(${TARGET_NAME}
                               PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED)
    set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES gfx1201)
  else()
    message(
      FATAL_ERROR
        "Unsupported backend '${BACKEND}' provided to addFfxExecutable.")
  endif()

  set_target_properties(
    ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                              OUTPUT_NAME ${TARGET_NAME})
endfunction()
