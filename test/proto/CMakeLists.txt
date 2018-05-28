proto_library(addressbook SRCS addressbook.proto)
cc_binary(add_person SRCS add_person.cc DEPS addressbook)


py_testing(test_add_person SRCS add_person.py ARGS addr_proto.bin)