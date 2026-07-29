get_filename_component(TEST_NAME ${MAIN_CC} NAME_WE)
set(TARGET_NAME "${TEST_NAME}_tbb")
add_executable(${TARGET_NAME} ${MAIN_CC})

target_compile_definitions(
  ${TARGET_NAME} PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
                         ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT=64)

target_link_libraries(${TARGET_NAME} PRIVATE ffx::ffx TBB::tbb)

if(TARGET model_data)
  target_link_libraries(${TARGET_NAME} PRIVATE model_data)
endif()

set_target_properties(
  ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                            OUTPUT_NAME ${TARGET_NAME})
