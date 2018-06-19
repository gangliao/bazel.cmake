cc_library(cpu_id SRCS cpu_id.cc)
cc_test(cpu_id_test SRCS cpu_id_test.cc DEPS cpu_id glog gtest)

cc_test(hello SRCS hello.cc)
