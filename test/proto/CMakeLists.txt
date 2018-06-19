# generate protobuf code
# 1. addressbook_pb2.py
# 2. addressbook_pb.cc -> libaddressbook.a 
proto_library(addressbook SRCS addressbook.proto)

cc_binary(add_person SRCS add_person.cc DEPS addressbook)
py_test(test_add_person SRCS add_person.py ARGS addr_proto.bin DEPS addressbook)
