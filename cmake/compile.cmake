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

if(MSVC)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D GFLAGS_DLL_DECLARE_FLAG=")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D GFLAGS_DLL_DEFINE_FLAG=")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D GFLAGS_IS_A_DLL=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D GLOG_NO_ABBREVIATED_SEVERITIES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D GOOGLE_GLOG_DLL_DECL=")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
    if(WITH_MSVC_MT)
      foreach(flag_var
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
          string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
      endforeach(flag_var)
    endif()
else(MSVC)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-std=c++11"    SUPPORT_CXX11)
    set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -fPIC")
endif(MSVC)
