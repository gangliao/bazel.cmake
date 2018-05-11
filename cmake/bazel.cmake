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


########################################################################
#
# bazel.cmake mimics the behavior of bazel (https://bazel.build/) to 
# simplify the usability of CMake.
#
# The [README.md] contains more information about how to use it.
#
########################################################################

if(NOT CMAKE_CROSSCOMPILING)
    find_package(CUDA QUIET)
endif(NOT CMAKE_CROSSCOMPILING)

option(WITH_GPU     "Compile Source Code with NVIDIA GPU"     ${CUDA_FOUND})
option(WITH_TESTING "Compile Source Code with Unit Testing"   OFF)

include(merge_libs)
include(external/cuda)
include(external/gflags)
include(external/glog)
include(external/gtest)

# including binary directory for generated headers.
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(NOT APPLE AND NOT ANDROID)
    find_package(Threads REQUIRED)
    link_libraries(${CMAKE_THREAD_LIBS_INIT})
    set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl -lrt")
endif(NOT APPLE AND NOT ANDROID)

function(cmake_library TARGET_NAME)
  set(options STATIC SHARED)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS EXTRA_DEPS)
  cmake_parse_arguments(cmake_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if (cmake_library_SRCS)
    if (cmake_library_SHARED) # build *.so
      set(_lib_type SHARED)
    else(cmake_library_SHARED)
      set(_lib_type STATIC)
    endif(cmake_library_SHARED)
    ${cmake_library_FUNC}(${TARGET_NAME} ${_lib_type} ${cmake_library_SRCS}) 
    if (cmake_library_DEPS)
      add_dependencies(${TARGET_NAME} ${cmake_library_DEPS} ${cmake_library_EXTRA_DEPS})
      target_link_libraries(${TARGET_NAME} ${cmake_library_DEPS} ${cmake_library_EXTRA_DEPS})
    endif(cmake_library_DEPS)
    if (cmake_library_EXTRA_DEPS)
      add_test(${TARGET_NAME} ${TARGET_NAME})
    endif(cmake_library_EXTRA_DEPS)
  else(cmake_library_SRCS)
    if (cmake_library_DEPS AND NOT ${cmake_library_FUNC} MATCHES "^add_executable$")
      merge_static_libs(${TARGET_NAME} ${cmake_library_DEPS})
    else(cmake_library_DEPS)
      message(FATAL "Please specify source files or libraries in CMake function.")
    endif(cmake_library_DEPS)
  endif(cmake_library_SRCS)
endfunction(cmake_library)

function(cc_library)
  cmake_library(${ARGV} "FUNC add_library")
endfunction(cc_library)

function(cc_binary)
  cmake_library(${ARGV} "FUNC add_executable")
endfunction(cc_binary)

function(cc_testing)
  cmake_library(${ARGV} "FUNC add_executable EXTRA_DEPS gtest gtest_main")
endfunction(cc_testing)

function(nv_library)
  cmake_library(${ARGV} "FUNC cuda_add_library")
endfunction(nv_library)

function(nv_binary)
  cmake_library(${ARGV} "FUNC cuda_add_executable")
endfunction(nv_binary)

function(nv_testing)
  cmake_library(${ARGV} "FUNC cuda_add_executable EXTRA_DEPS gtest gtest_main")
endfunction(nv_testing)
