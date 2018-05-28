cc_library(cpu_id SRCS cpu_id.cc)
cc_testing(cpu_id_test SRCS cpu_id_test.cc DEPS cpu_id glog)

cc_binary(hello SRCS hello.cc)