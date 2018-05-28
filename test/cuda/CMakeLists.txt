nv_library(vector_add SRCS vector_add.cu)
nv_testing(vector_add_test SRCS vector_add_test.cu DEPS vector_add)
