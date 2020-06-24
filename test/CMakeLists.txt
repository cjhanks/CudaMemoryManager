enable_testing()
find_package(GTest)

if (${GTEST_FOUND})
  file(GLOB_RECURSE SOURCES_CC  *.cc)
  file(GLOB_RECURSE SOURCES_CU  *.cu)

  add_executable(test_cmm ${SOURCES_CC} ${SOURCES_CU})
  target_link_libraries(test_cmm GTest::GTest GTest::Main cmm)
  add_test(Tests test_cmm)
endif()

