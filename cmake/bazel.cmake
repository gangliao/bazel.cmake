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

function(merge_static_libs TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)

  # Get all propagation dependencies from the merged libraries
  foreach(lib ${libs})
    list(APPEND libs_deps ${${lib}_LIB_DEPENDS})
  endforeach()
  list(REMOVE_DUPLICATES libs_deps)

  if(APPLE) # Use OSX's libtool to merge archives
    # To produce a library we need at least one source file,
    # which is created by add_custom_command below. It
    # also can help to track dependencies.
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)

    # Make the generated dummy source file dependeds on all static input
    # libraries. If they are changed, the dummy file is also touched
    # which causes the desired effect (relink).
    add_custom_command(OUTPUT ${dummyfile}
      COMMAND ${CMAKE_COMMAND} -E touch ${dummyfile}
      DEPENDS ${libs})

    # Generate dummy staic library
    file(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")
    add_library(${TARGET_NAME} STATIC ${dummyfile})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    foreach(lib ${libs})
      # Get the filenames of the merged libraries 
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND rm -rf "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a"
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
