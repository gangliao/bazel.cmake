# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

INCLUDE(ExternalProject)

# Always invoke `FIND_PACKAGE(Protobuf)` for importing function protobuf_generate_cpp
FIND_PACKAGE(Protobuf QUIET)

function(_protobuf_generate_cpp SRCS HDRS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: _protobuf_generate_cpp() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})

  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    
    set(_protobuf_protoc_src "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    set(_protobuf_protoc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")
    list(APPEND ${SRCS} "${_protobuf_protoc_src}")
    list(APPEND ${HDRS} "${_protobuf_protoc_hdr}")
    
    add_custom_command(
      OUTPUT "${_protobuf_protoc_src}"
             "${_protobuf_protoc_hdr}"

      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} 
      -I${CMAKE_CURRENT_SOURCE_DIR}
      --cpp_out "${CMAKE_CURRENT_BINARY_DIR}" ${ABS_FIL}
      DEPENDS ${ABS_FIL} protoc
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

function(_protobuf_generate_python SRCS)
    # shameless copy from https://github.com/Kitware/CMake/blob/master/Modules/FindProtobuf.cmake
    if(NOT ARGN)
        message(SEND_ERROR "Error: PROTOBUF_GENERATE_PYTHON() called without any proto files")
        return()
    endif()

    if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
        # Create an include path for each file specified
        foreach(FIL ${ARGN})
            get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
            get_filename_component(ABS_PATH ${ABS_FIL} PATH)
            list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
            if(${_contains_already} EQUAL -1)
                list(APPEND _protobuf_include_path -I ${ABS_PATH})
            endif()
        endforeach()
    else()
        set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
        set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
    endif()

    if(DEFINED Protobuf_IMPORT_DIRS)
        foreach(DIR ${Protobuf_IMPORT_DIRS})
            get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
            list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
            if(${_contains_already} EQUAL -1)
                list(APPEND _protobuf_include_path -I ${ABS_PATH})
            endif()
        endforeach()
    endif()

    set(${SRCS})
    foreach(FIL ${ARGN})
        get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
        get_filename_component(FIL_WE ${FIL} NAME_WE)
        if(NOT PROTOBUF_GENERATE_CPP_APPEND_PATH)
            get_filename_component(FIL_DIR ${FIL} DIRECTORY)
            if(FIL_DIR)
                set(FIL_WE "${FIL_DIR}/${FIL_WE}")
            endif()
        endif()

        list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py")
        add_custom_command(
                OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py"
                COMMAND  ${Protobuf_PROTOC_EXECUTABLE} --python_out ${CMAKE_CURRENT_BINARY_DIR} ${_protobuf_include_path} ${ABS_FIL}
                DEPENDS ${ABS_FIL} ${Protobuf_PROTOC_EXECUTABLE}
                COMMENT "Running Python protocol buffer compiler on ${FIL}"
                VERBATIM )
    endforeach()

    set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()

# Print and set the protobuf library information,
# finish this cmake process and exit from this file.
macro(PROMPT_PROTOBUF_LIB)
    SET(protobuf_DEPS ${ARGN})

    MESSAGE(STATUS "Google/protobuf executable: ${PROTOBUF_PROTOC_EXECUTABLE}")
    MESSAGE(STATUS "Google/protobuf library: ${PROTOBUF_LIBRARY}")
    MESSAGE(STATUS "Google/protobuf version: ${PROTOBUF_VERSION}")
    INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})

    # Assuming that all the protobuf libraries are of the same type.
    IF(${PROTOBUF_LIBRARY} MATCHES "${CMAKE_STATIC_LIBRARY_SUFFIX}$")
        SET(protobuf_LIBTYPE STATIC)
    ELSEIF(${PROTOBUF_LIBRARY} MATCHES "${CMAKE_SHARED_LIBRARY_SUFFIX}$")
        SET(protobuf_LIBTYPE SHARED)
    ELSE()
        MESSAGE(FATAL_ERROR "Unknown library type: ${PROTOBUF_LIBRARY}")
    ENDIF()

    ADD_LIBRARY(protobuf ${protobuf_LIBTYPE} IMPORTED GLOBAL)
    SET_PROPERTY(TARGET protobuf PROPERTY IMPORTED_LOCATION ${PROTOBUF_LIBRARY})

    ADD_LIBRARY(protobuf_lite ${protobuf_LIBTYPE} IMPORTED GLOBAL)
    SET_PROPERTY(TARGET protobuf_lite PROPERTY IMPORTED_LOCATION ${PROTOBUF_LITE_LIBRARY})

    ADD_LIBRARY(libprotoc ${protobuf_LIBTYPE} IMPORTED GLOBAL)
    SET_PROPERTY(TARGET libprotoc PROPERTY IMPORTED_LOCATION ${PROTOC_LIBRARY})

    ADD_EXECUTABLE(protoc IMPORTED GLOBAL)
    SET_PROPERTY(TARGET protoc PROPERTY IMPORTED_LOCATION ${PROTOBUF_PROTOC_EXECUTABLE})
    # FIND_Protobuf.cmake uses `Protobuf_PROTOC_EXECUTABLE`.
    # make `protobuf_generate_cpp` happy.
    SET(Protobuf_PROTOC_EXECUTABLE ${PROTOBUF_PROTOC_EXECUTABLE})
    FOREACH(dep ${protobuf_DEPS})
        ADD_DEPENDENCIES(protobuf ${dep})
        ADD_DEPENDENCIES(protobuf_lite ${dep})
        ADD_DEPENDENCIES(libprotoc ${dep})
        ADD_DEPENDENCIES(protoc ${dep})
    ENDFOREACH()

    RETURN()
endmacro()

macro(SET_PROTOBUF_VERSION)
    EXEC_PROGRAM(${PROTOBUF_PROTOC_EXECUTABLE} ARGS --version OUTPUT_VARIABLE PROTOBUF_VERSION)
    STRING(REGEX MATCH "[0-9]+.[0-9]+" PROTOBUF_VERSION "${PROTOBUF_VERSION}")
endmacro()

set(PROTOBUF_ROOT "" CACHE PATH "Folder contains protobuf")
if (NOT "${PROTOBUF_ROOT}" STREQUAL "")
    find_path(PROTOBUF_INCLUDE_DIR google/protobuf/message.h PATHS ${PROTOBUF_ROOT}/include NO_DEFAULT_PATH)
    find_library(PROTOBUF_LIBRARY protobuf PATHS ${PROTOBUF_ROOT}/lib NO_DEFAULT_PATH)
    find_library(PROTOBUF_LITE_LIBRARY protobuf-lite PATHS ${PROTOBUF_ROOT}/lib NO_DEFAULT_PATH)
    find_library(PROTOBUF_PROTOC_LIBRARY protoc PATHS ${PROTOBUF_ROOT}/lib NO_DEFAULT_PATH)
    find_program(PROTOBUF_PROTOC_EXECUTABLE protoc PATHS ${PROTOBUF_ROOT}/bin NO_DEFAULT_PATH)
    if (PROTOBUF_INCLUDE_DIR AND PROTOBUF_LIBRARY AND PROTOBUF_LITE_LIBRARY AND PROTOBUF_PROTOC_LIBRARY AND PROTOBUF_PROTOC_EXECUTABLE)
        message(STATUS "Using custom protobuf library in ${PROTOBUF_ROOT}.")
        SET_PROTOBUF_VERSION()
        PROMPT_PROTOBUF_LIB()
    else()
        message(WARNING "Cannot find protobuf library in ${PROTOBUF_ROOT}.")
    endif()
endif()

FUNCTION(build_protobuf TARGET_NAME BUILD_FOR_HOST)
    STRING(REPLACE "extern_" "" TARGET_DIR_NAME "${TARGET_NAME}")
    STRING(REPLACE "_host" "" TARGET_DIR_NAME "${TARGET_DIR_NAME}")
    SET(PROTOBUF_SOURCES_DIR ${BAZEL_THIRD_PARTY_DIR}/${TARGET_DIR_NAME})
    SET(PROTOBUF_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/${TARGET_NAME})

    SET(${TARGET_NAME}_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(PROTOBUF_INCLUDE_DIR "${PROTOBUF_INSTALL_DIR}/include" PARENT_SCOPE)
    SET(${TARGET_NAME}_LITE_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite${CMAKE_STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_LIBRARY
        "${PROTOBUF_INSTALL_DIR}/lib/libprotoc${CMAKE_STATIC_LIBRARY_SUFFIX}"
         PARENT_SCOPE)
    SET(${TARGET_NAME}_PROTOC_EXECUTABLE
        "${PROTOBUF_INSTALL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}"
         PARENT_SCOPE)

    SET(OPTIONAL_CACHE_ARGS "")
    SET(OPTIONAL_ARGS "")
    IF(BUILD_FOR_HOST)
        SET(OPTIONAL_ARGS
            "-DCMAKE_CXX_COMPILER=${HOST_CXX_COMPILER}"
            "-DCMAKE_C_COMPILER=${HOST_C_COMPILER}"
            "-Dprotobuf_WITH_ZLIB=OFF")
        SET(EXTERNAL_PROJECT_CMAKE_ARGS "")
    ELSE()
        SET(OPTIONAL_ARGS
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
            "-DCMAKE_AR=${CMAKE_AR}"
            "-DCMAKE_RANLIB=${CMAKE_RANLIB}"
            "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
            "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
        SET(OPTIONAL_CACHE_ARGS
            "-Dprotobuf_WITH_ZLIB:BOOL=ON"
            "-DZLIB_FOUND:BOOL=ON"
            "-DZLIB_INCLUDE_DIRS:PATH=${ZLIB_INCLUDE_DIR}"
            "-DZLIB_LIBRARIES:FILEPATH=${ZLIB_LIBRARIES}"
            "-Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF")
        IF(CMAKE_CROSSCOMPILING)
            SET(OPTIONAL_ARGS ${OPTIONAL_ARGS} "-Dprotobuf_BUILD_PROTOC_BINARIES:BOOL=OFF")
            SET(OPTIONAL_CACHE_ARGS ${OPTIONAL_CACHE_ARGS} "-Dprotobuf_BUILD_PROTOC_BINARIES:BOOL=OFF")
        ENDIF()
    ENDIF()

    ExternalProject_Add(
        ${TARGET_NAME}
        ${EXTERNAL_PROJECT_LOG_ARGS}
        SOURCE_DIR       ${PROTOBUF_SOURCES_DIR}
        UPDATE_COMMAND   ""
        DOWNLOAD_COMMAND ""
        ${EXTERNAL_PROJECT_CMAKE_ARGS}
        DEPENDS          zlib
        CONFIGURE_COMMAND
        ${CMAKE_COMMAND} ${PROTOBUF_SOURCES_DIR}/cmake
            -Dprotobuf_BUILD_TESTS=OFF
            -DCMAKE_SKIP_RPATH=ON
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
            -DCMAKE_INSTALL_LIBDIR=lib
            ${OPTIONAL_ARGS}
        CMAKE_CACHE_ARGS
            -DCMAKE_INSTALL_PREFIX:PATH=${PROTOBUF_INSTALL_DIR}
            -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
            ${OPTIONAL_CACHE_ARGS}
    )
ENDFUNCTION()


IF(CMAKE_CROSSCOMPILING)
    build_protobuf(protobuf_host TRUE)
    SET(PROTOBUF_PROTOC_EXECUTABLE ${protobuf_host_PROTOC_EXECUTABLE}
        CACHE FILEPATH "protobuf executable." FORCE)
    SET_PROTOBUF_VERSION()
ENDIF()

IF(NOT PROTOBUF_FOUND)
    build_protobuf(extern_protobuf FALSE)
    ADD_DEPENDENCIES(extern_protobuf protobuf_host)

    SET(PROTOBUF_INCLUDE_DIR ${extern_protobuf_INCLUDE_DIR}
        CACHE PATH "protobuf include directory." FORCE)
    SET(PROTOBUF_LITE_LIBRARY ${extern_protobuf_LITE_LIBRARY}
        CACHE FILEPATH "protobuf lite library." FORCE)
    SET(PROTOBUF_LIBRARY ${extern_protobuf_LIBRARY}
        CACHE FILEPATH "protobuf library." FORCE)
    SET(PROTOBUF_PROTOC_LIBRARY ${extern_protobuf_PROTOC_LIBRARY}
        CACHE FILEPATH "protoc library." FORCE)

    IF(CMAKE_CROSSCOMPILING)
        PROMPT_PROTOBUF_LIB(protobuf_host extern_protobuf)
    ELSE()
        SET(PROTOBUF_PROTOC_EXECUTABLE ${extern_protobuf_PROTOC_EXECUTABLE}
            CACHE FILEPATH "protobuf executable." FORCE)
        SET_PROTOBUF_VERSION()
        PROMPT_PROTOBUF_LIB(extern_protobuf)
    ENDIF()
ENDIF(NOT PROTOBUF_FOUND)
