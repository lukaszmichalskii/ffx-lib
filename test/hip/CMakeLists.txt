enable_language(HIP)

foreach(SOURCE_FILE ${SOURCE_FILES})
  addffxtest(${SOURCE_FILE} "hip")
endforeach()
