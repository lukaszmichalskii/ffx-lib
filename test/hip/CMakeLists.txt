enable_language(HIP)

foreach(SOURCE_FILE ${SOURCE_FILES})
  ffx_add_test(${SOURCE_FILE} "hip")
endforeach()
