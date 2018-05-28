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

IF(NOT WITH_PYTHON)
    return()
ENDIF()

FIND_PACKAGE(PythonInterp   REQUIRED)
FIND_PACKAGE(PythonLibs     REQUIRED)

# Fixme: Maybe find a static library. Get SHARED/STATIC by FIND_PACKAGE.
ADD_LIBRARY(python SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET python PROPERTY IMPORTED_LOCATION ${PYTHON_LIBRARIES})

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIR})
MESSAGE(STATUS "Python executable: ${PYTHON_EXECUTABLE}")
MESSAGE(STATUS "Python library: ${PYTHON_LIBRARIES}")
MESSAGE(STATUS "Python include: ${PYTHON_INCLUDE_DIR}")
