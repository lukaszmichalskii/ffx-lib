get_filename_component(TEST_NAME ${MAIN_CC} NAME_WE)
set(TARGET_NAME "${TEST_NAME}_hip")

set_source_files_properties(${MAIN_CC} PROPERTIES LANGUAGE HIP)

add_executable(${TARGET_NAME} ${MAIN_CC})

target_compile_definitions(${TARGET_NAME} PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED)

target_link_libraries(${TARGET_NAME} PRIVATE ffx::ffx)

if(TARGET model_data)
  target_link_libraries(${TARGET_NAME} PRIVATE model_data)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES gfx1201)
set_target_properties(
  ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                            OUTPUT_NAME ${TARGET_NAME})
