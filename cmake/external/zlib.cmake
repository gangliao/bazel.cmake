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

INCLUDE(ExternalProject)

SET(ZLIB_SOURCES_DIR ${BAZEL_THIRD_PARTY_DIR}/zlib)
SET(ZLIB_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib)
SET(ZLIB_ROOT ${ZLIB_INSTALL_DIR} CACHE FILEPATH "zlib root directory." FORCE)
SET(ZLIB_INCLUDE_DIR "${ZLIB_INSTALL_DIR}/include" CACHE PATH "zlib include directory." FORCE)

IF(WIN32)
  SET(ZLIB_LIBRARIES "${ZLIB_INSTALL_DIR}/lib/zlibstatic.lib" CACHE FILEPATH "zlib library." FORCE)
ELSE(WIN32)
  SET(ZLIB_LIBRARIES "${ZLIB_INSTALL_DIR}/lib/libz.a" CACHE FILEPATH "zlib library." FORCE)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR})

ExternalProject_Add(
    extern_zlib
    ${EXTERNAL_PROJECT_LOG_ARGS}
    SOURCE_DIR       ${ZLIB_SOURCES_DIR}
    GIT_TAG          "v1.2.8"
    UPDATE_COMMAND   ""
    DOWNLOAD_COMMAND ""
    ${EXTERNAL_PROJECT_CMAKE_ARGS}
    CMAKE_ARGS      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                    -DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR}
                    -DBUILD_SHARED_LIBS=OFF
                    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    -DCMAKE_MACOSX_RPATH=ON
                    ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${ZLIB_INSTALL_DIR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
)

ADD_LIBRARY(zlib STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET zlib PROPERTY IMPORTED_LOCATION ${ZLIB_LIBRARIES})
ADD_DEPENDENCIES(zlib extern_zlib)
