# Copyright (c) 2016 Gang Liao. All Rights Reserve.
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
#
# Gang Liao, liao.gang@kaust.edu.sa
#
# generic.cmake defines CMakes functions that look like Bazel's
# building rules (https://bazel.build/).
#
#
# -------------------------------------------
#     C++        CUDA C++
# -------------------------------------------
# cc_library    nv_library
# cc_testing    nv_testing
# -------------------------------------------
#
# To build a static library example.a from example.cc using the system
#  compiler (like GCC):
#
#   cc_library(example SRCS example.cc)
#
# To build a static library example.a from multiple source files
# example{1,2,3}.cc:
#
#   cc_library(example SRCS example1.cc example2.cc example3.cc)
#
# To build a shared library example.so from example.cc:
#
#   cc_library(example SHARED SRCS example.cc)
#
# To build a library using Nvidia's NVCC from .cu file(s), use the nv_
# prefixed version:
#
#   nv_library(example SRCS example.cu)
#
# To specify that a library new_example.a depends on other libraies:
#
#   cc_library(new_example SRCS new_example.cc DEPS example)
#
# Static libraries can be composed of other static libraries:
#
#   cc_library(composed DEPS dependent1 dependent2 dependent3)
#
# To build a unit test binary, which is an executable binary with
# GoogleTest linked:
#
#   cc_testing(example_test SRCS example_test.cc DEPS example)
#
# To build a unit test binary using NVCC, use the nv_ prefixed version:
#
#   nv_testing(example_test SRCS example_test.cu DEPS example)
#
# It is pretty often that executable and test binaries depend on
# pre-defined external libaries like glog and gflags defined in
# /cmake/external/*.cmake:
#
#   cc_testing(example_test SRCS example_test.cc DEPS example glog gflags)
#

# including binary directory for generated headers.
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(NOT APPLE AND NOT ANDROID)
    find_package(Threads REQUIRED)
    link_libraries(${CMAKE_THREAD_LIBS_INIT})
    set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl -lrt")
endif(NOT APPLE AND NOT ANDROID)

function(merge_static_libs TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)

  # Get all propagation dependencies from the merged libraries
  foreach(lib ${libs})
    list(APPEND libs_deps ${${lib}_LIB_DEPENDS})
  endforeach()
  list(REMOVE_DUPLICATES libs_deps)

  if(APPLE) # Use OSX's libtool to merge archives
    # To produce a library we need at least one source file.
    # It is created by add_custom_command below and will helps
    # also help to track dependencies.
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)

    # Make the generated dummy source file depended on all static input
    # libs. If input lib changes,the source file is touched
    # which causes the desired effect (relink).
    add_custom_command(OUTPUT ${dummyfile}
      COMMAND ${CMAKE_COMMAND} -E touch ${dummyfile}
      DEPENDS ${libs})

    # Generate dummy staic lib
    file(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")
    add_library(${TARGET_NAME} STATIC ${dummyfile})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    foreach(lib ${libs})
      # Get the file names of the libraries to be merged
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND rm "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a"
      COMMAND /usr/bin/libtool -static -o "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a" ${libfiles})
  else() # general UNIX: use "ar" to extract objects and re-add to a common lib
    foreach(lib ${libs})
      set(objlistfile ${lib}.objlist) # list of objects in the input library
      set(objdir ${lib}.objdir)

      add_custom_command(OUTPUT ${objdir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${objdir}
        DEPENDS ${lib})

      add_custom_command(OUTPUT ${objlistfile}
        COMMAND ${CMAKE_AR} -x "$<TARGET_FILE:${lib}>"
        COMMAND ${CMAKE_AR} -t "$<TARGET_FILE:${lib}>" > ../${objlistfile}
        DEPENDS ${lib} ${objdir}
        WORKING_DIRECTORY ${objdir})

      # Empty dummy source file that goes into merged library		
      set(mergebase ${lib}.mergebase.c)		
      add_custom_command(OUTPUT ${mergebase}		
        COMMAND ${CMAKE_COMMAND} -E touch ${mergebase}		
        DEPENDS ${objlistfile})		

      list(APPEND mergebases "${mergebase}")
    endforeach()

    add_library(${TARGET_NAME} STATIC ${mergebases})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    # Get the file name of the generated library
    set(outlibfile "$<TARGET_FILE:${TARGET_NAME}>")

    foreach(lib ${libs})
      add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_AR} cr ${outlibfile} *.o
        COMMAND ${CMAKE_RANLIB} ${outlibfile}
        WORKING_DIRECTORY ${lib}.objdir)
    endforeach()
  endif()
endfunction(merge_static_libs)

function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if (cc_library_SRCS)
    if (cc_library_SHARED OR cc_library_shared) # build *.so
      add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
    endif()
    if (cc_library_DEPS)
      add_dependencies(${TARGET_NAME} ${cc_library_DEPS})
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    endif()
  else(cc_library_SRCS)
    if (cc_library_DEPS)
      merge_static_libs(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL "Please specify source file or library in cc_library.")
    endif()
  endif(cc_library_SRCS)
endfunction(cc_library)

function(cc_testing TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_testing "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_executable(${TARGET_NAME}_test ${cc_testing_SRCS})
  target_link_libraries(${TARGET_NAME}_test ${cc_testing_DEPS} gtest gtest_main)
  add_dependencies(${TARGET_NAME}_test ${cc_testing_DEPS} gtest gtest_main)
  add_test(${TARGET_NAME}_test ${TARGET_NAME}_test)
endfunction(cc_testing)

function(nv_library TARGET_NAME)
  if (WITH_GPU)
    set(options STATIC static SHARED shared)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(nv_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(nv_library_SRCS)
      if (nv_library_SHARED OR nv_library_shared) # build *.so
        cuda_add_library(${TARGET_NAME} SHARED ${nv_library_SRCS})
      else()
          cuda_add_library(${TARGET_NAME} STATIC ${nv_library_SRCS})
      endif()
      if (nv_library_DEPS)
        add_dependencies(${TARGET_NAME} ${nv_library_DEPS})
        target_link_libraries(${TARGET_NAME} ${nv_library_DEPS})
      endif()
    else(nv_library_SRCS)
      if (nv_library_DEPS)
        merge_static_libs(${TARGET_NAME} ${nv_library_DEPS})
      else()
        message(FATAL "Please specify source file or library in nv_library.")
      endif()
    endif(nv_library_SRCS)
  endif()
endfunction(nv_library)

function(nv_testing TARGET_NAME)
  if (WITH_GPU)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(nv_testing "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cuda_add_executable(${TARGET_NAME}_test ${nv_testing_SRCS})
    target_link_libraries(${TARGET_NAME}_test ${nv_testing_DEPS} gtest gtest_main)
    add_dependencies(${TARGET_NAME}_test ${nv_testing_DEPS} gtest gtest_main)
    add_test(${TARGET_NAME}_test ${TARGET_NAME}_test)
  endif()
endfunction(nv_testing)
