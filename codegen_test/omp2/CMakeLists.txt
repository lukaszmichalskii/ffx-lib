get_filename_component(TEST_NAME ${MAIN_CC} NAME_WE)
set(TARGET_NAME "${TEST_NAME}_omp2")
add_executable(${TARGET_NAME} ${MAIN_CC})

target_compile_definitions(
  ${TARGET_NAME} PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
                         ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=64)

target_link_libraries(${TARGET_NAME} PRIVATE ffx::ffx OpenMP::OpenMP_CXX)

if(TARGET model_data)
  target_link_libraries(${TARGET_NAME} PRIVATE model_data)
endif()

set_target_properties(
  ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                            OUTPUT_NAME ${TARGET_NAME})
