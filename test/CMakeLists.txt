# Copyright (c) 2016 Gang Liao <gangliao@umd.edu> All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(testcase VERSION 0.1.0)

set(TESTCASE_SOURCE_DIR   ${CMAKE_CURRENT_SOURCE_DIR})
set(TESTCASE_BINARY_DIR   ${CMAKE_CURRENT_BINARY_DIR})

get_filename_component(cur_root_dir ${TESTCASE_SOURCE_DIR} DIRECTORY)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${cur_root_dir}/cmake)

include(bazel)

add_subdirectory(c++)
add_subdirectory(cuda)
add_subdirectory(proto)