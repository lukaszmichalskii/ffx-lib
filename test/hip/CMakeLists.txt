enable_language(HIP)

foreach(SOURCE_FILE ${PORTABLE_SOURCES})
  ffx_add_test(${SOURCE_FILE} "hip")
endforeach()

foreach(SOURCE_FILE ${HIP_SOURCES})
  ffx_add_test(${SOURCE_FILE} "hip")
endforeach()
