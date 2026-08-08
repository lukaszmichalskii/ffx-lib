foreach(SOURCE_FILE ${PORTABLE_SOURCES})
  ffx_add_test(${SOURCE_FILE} "tbb")
endforeach()
