set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

get_filename_component(TEST_NAME ${MAIN_CC} NAME_WE)
set(TARGET_NAME "${TEST_NAME}_cuda")
set_source_files_properties(${MAIN_CC} PROPERTIES LANGUAGE CUDA)
add_executable(${TARGET_NAME} ${MAIN_CC})

target_compile_definitions(${TARGET_NAME} PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)

target_link_libraries(${TARGET_NAME} PRIVATE ffx::ffx)

if(TARGET model_data)
  target_link_libraries(${TARGET_NAME} PRIVATE model_data)
endif()

target_compile_options(
  ${TARGET_NAME} PRIVATE --expt-relaxed-constexpr --expt-extended-lambda
                         -Xcompiler -std=c++23)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                                CUDA_ARCHITECTURES 120)
set_target_properties(
  ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                            OUTPUT_NAME ${TARGET_NAME})
