foreach(SOURCE_FILE ${SOURCE_FILES})
  ffx_add_test(${SOURCE_FILE} "tbb")
endforeach()
