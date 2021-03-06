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

set(TEND_TO_USE_GPU OFF)
if(NOT CMAKE_CROSSCOMPILING)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        set(TEND_TO_USE_GPU ON)
    endif(CUDA_FOUND)
endif(NOT CMAKE_CROSSCOMPILING)

option(WITH_GPU     "Compile Source Code with NVIDIA GPU"     ${TEND_TO_USE_GPU})
option(WITH_TESTING "Compile Source Code with Unit Testing"   ON)
option(WITH_PYTHON  "Compile Source Code with Python"         ON)  
option(WITH_MSVC_MT "Compile Source Code with MSVC /MT"       OFF)

message(STATUS "CUDA_FOUND=" ${CUDA_FOUND})
message(STATUS "WITH_GPU=" ${WITH_GPU})
message(STATUS "WITH_TESTING=" ${WITH_TESTING})
message(STATUS "WITH_PYTHON=" ${WITH_PYTHON})
message(STATUS "WITH_MSVC_MT=" ${WITH_MSVC_MT})

get_filename_component(BAZEL_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
set(BAZEL_THIRD_PARTY_DIR ${BAZEL_SOURCE_DIR}/third-party)

include(color)
if(CMAKE_CROSSCOMPILING)
  include(host)
endif(CMAKE_CROSSCOMPILING)
include(compile)
include(merge_libs)
include(external/cuda)
include(external/gflags)
include(external/glog)
include(external/gtest)
include(external/python)
include(external/zlib)
include(external/protobuf)

add_custom_target(helps COMMAND ./make_help WORKING_DIRECTORY ${BAZEL_SOURCE_DIR}/cmake)

macro(_build_target func_tag)
  set(_sources ${ARGN})

  # Given a variable containing a file list,
  # it will remove all the files wich basename
  # does not match the specified pattern.
  if(${CMAKE_VERSION} VERSION_LESS 3.6)
    foreach(src_file ${_sources})
      get_filename_component(base_name ${src_file} NAME)
      if(${base_name} MATCHES "\\.proto$")
        list(REMOVE_ITEM _sources "${src_file}")
      endif()
    endforeach()
  else(${CMAKE_VERSION} VERSION_LESS 3.6)
    list(FILTER _sources EXCLUDE REGEX "\\.proto$")
  endif(${CMAKE_VERSION} VERSION_LESS 3.6)

  if (${func_tag} STREQUAL "cc_lib")
    add_library(${_sources})
  elseif(${func_tag} STREQUAL "cc_bin")
    list(REMOVE_ITEM _sources STATIC SHARED)
    add_executable(${_sources})
  elseif(${func_tag} STREQUAL "nv_lib")
    cuda_add_library(${_sources})
  elseif(${func_tag} STREQUAL "nv_bin")
    list(REMOVE_ITEM _sources STATIC SHARED)
    cuda_add_executable(${_sources})
  endif()
endmacro(_build_target)

function(cmake_library TARGET_NAME)
  set(options STATIC SHARED)
  set(oneValueArgs TAG)
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cmake_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if (cmake_library_SRCS)
    if (cmake_library_SHARED) # build *.so
      set(_lib_type SHARED)
    else(cmake_library_SHARED)
      set(_lib_type STATIC)
    endif(cmake_library_SHARED)
    _build_target(${cmake_library_TAG} ${TARGET_NAME} ${_lib_type} ${cmake_library_SRCS}) 
    if (cmake_library_DEPS)
      add_dependencies(${TARGET_NAME} ${cmake_library_DEPS})
      target_link_libraries(${TARGET_NAME} ${cmake_library_DEPS})
    endif(cmake_library_DEPS)
  else(cmake_library_SRCS)
    if (cmake_library_DEPS AND ${cmake_library_TAG} STREQUAL "cc_lib")
      merge_static_libs(${TARGET_NAME} ${cmake_library_DEPS})
    else()
      message(FATAL_ERROR "Please review the valid syntax: typing `make helps` in the Terminal"
                          "or visiting https://github.com/gangliao/bazel.cmake#cheat-sheet")
    endif()
  endif(cmake_library_SRCS)
endfunction(cmake_library)

macro(check_gtest)
  set(options STATIC SHARED)
  set(oneValueArgs TAG)
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(check_gtest "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  set(gtest_deps DEPS gtest gtest_main)
  foreach(filename ${check_gtest_SRCS})
    file(STRINGS ${filename} output REGEX "(int|void)[ ]+main")
    if(output)
      list(REMOVE_ITEM gtest_deps gtest_main)
      break()
    endif(output)
  endforeach()

  set(${gtest_deps} ${gtest_deps} PARENT_SCOPE)
endmacro(check_gtest)

function(cc_library)
  cmake_library(${ARGV} TAG cc_lib)
endfunction(cc_library)

function(cc_binary)
  cmake_library(${ARGV} TAG cc_bin)
endfunction(cc_binary)

function(cc_test)
  check_gtest(${ARGV})
  cmake_library(${ARGV} TAG cc_bin ${gtest_deps})
  add_test(${ARGV0} ${ARGV0})
endfunction(cc_test)

function(nv_library)
  if (WITH_GPU)
    cmake_library(${ARGV} TAG nv_lib)
  endif(WITH_GPU)
endfunction(nv_library)

function(nv_binary)
  if (WITH_GPU)
    cmake_library(${ARGV} TAG nv_bin)
  endif(WITH_GPU)
endfunction(nv_binary)

function(nv_test)
  if (WITH_GPU)
    check_gtest(${ARGV})
    cmake_library(${ARGV} TAG nv_bin ${gtest_deps})
    add_test(${ARGV0} ${ARGV0})
  endif(WITH_GPU)
endfunction(nv_test)

function(proto_library)
  set(options STATIC SHARED)
  set(oneValueArgs TAG)
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(proto_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  _protobuf_generate_cpp(proto_srcs proto_hdrs ${proto_library_SRCS})
  _protobuf_generate_python(py_srcs ${proto_library_SRCS})

  # including binary directory for generated headers (protobuf hdrs).
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  cmake_library(${ARGV} SRCS ${proto_srcs} ${proto_hdrs} DEPS protobuf TAG cc_lib)

  add_custom_target(py_${ARGV0} ALL DEPENDS ${py_srcs})
  # Create __init__.py in all ancestor directories of where the .proto
  # files resides so to make the *_pb2.py a importable module.
  get_filename_component(cur_dir ${py_srcs} DIRECTORY)
  while(NOT ${cur_dir} STREQUAL ${CMAKE_BINARY_DIR})
    file(WRITE ${cur_dir}/__init__.py)
    get_filename_component(cur_dir ${cur_dir} DIRECTORY)
  endwhile()
endfunction(proto_library)

function(py_test)
  if (WITH_PYTHON)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    cmake_parse_arguments(py_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_test(NAME ${ARGV0}
      COMMAND env PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR} ${py_test_ENVS}
      ${PYTHON_EXECUTABLE} ${py_test_SRCS} ${py_test_ARGS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif(WITH_PYTHON)
endfunction()
