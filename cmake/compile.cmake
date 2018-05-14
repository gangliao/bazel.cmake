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

# external dependencies build args for mobile
IF(CMAKE_TOOLCHAIN_FILE AND IOS_PLATFORM)
        SET(EXTERNAL_PROJECT_CMAKE_ARGS
            CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            CMAKE_ARGS -DIOS_PLATFORM=${IOS_PLATFORM}
        )
        add_definitions(-D__arm__)
ELSE()
    SET(EXTERNAL_PROJECT_CMAKE_ARGS "")
ENDIF()

if(NOT APPLE AND NOT ANDROID)
    find_package(Threads REQUIRED)
    link_libraries(${CMAKE_THREAD_LIBS_INIT})
    set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl -lrt")
endif(NOT APPLE AND NOT ANDROID)

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


################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   detect_installed_gpus(out_variable)
function(detect_installed_gpus out_variable)
    if(NOT CUDA_gpu_detect_output)
        set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

        file(WRITE ${__cufile} ""
        "#include <cstdio>\n"
        "int main()\n"
        "{\n"
        "  int count = 0;\n"
        "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
        "  if (count == 0) return -1;\n"
        "  for (int device = 0; device < count; ++device)\n"
        "  {\n"
        "    cudaDeviceProp prop;\n"
        "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
        "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
        "  }\n"
        "  return 0;\n"
        "}\n")
        enable_language(CUDA)

        try_run(__nvcc_res __compile_result ${PROJECT_BINARY_DIR} ${__cufile}
        COMPILE_OUTPUT_VARIABLE __compile_out
        RUN_OUTPUT_VARIABLE __nvcc_out)

        if(__nvcc_res EQUAL 0 AND __compile_result)
            # nvcc outputs text containing line breaks when building with MSVC.
            # The line below prevents CMake from inserting a variable with line
            # breaks in the cache
            string(REGEX MATCH "([1-9].[0-9])" __nvcc_out "${__nvcc_out}")
            string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
            set(CUDA_gpu_detect_output ${__nvcc_out})
        else()
            message(WARNING "Running GPU detection script with nvcc failed: ${__nvcc_out} ${__compile_out}")
        endif()
    endif()

    if(NOT CUDA_gpu_detect_output)
        message(WARNING "Automatic GPU detection failed. Building for the default architectures.")
        set(${out_variable} "" PARENT_SCOPE)
    else()
        set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
    endif()
endfunction()

detect_installed_gpus(CUDA_NVCC_ARCH_FLAGS)
list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_ARCH_FLAGS} -Wno-deprecated-gpu-targets -std=c++11)