file(GLOB_RECURSE SOURCES_CC  *.cc)
file(GLOB_RECURSE SOURCES_CU  *.cu)

add_executable(cmm_example_ipc ${SOURCES_CC} ${SOURCES_CU})
target_link_libraries(cmm_example_ipc cmm)
