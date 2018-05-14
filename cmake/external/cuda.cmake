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

if(WITH_GPU)
    ADD_LIBRARY(cudart SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET cudart PROPERTY IMPORTED_LOCATION ${CUDA_LIBRARIES})

    ADD_LIBRARY(curand SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET curand PROPERTY IMPORTED_LOCATION ${CUDA_curand_LIBRARY})

    ADD_LIBRARY(cublas SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET cublas PROPERTY IMPORTED_LOCATION ${CUDA_CUBLAS_LIBRARIES})

    ADD_LIBRARY(cusparse SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET cusparse PROPERTY IMPORTED_LOCATION ${CUDA_cusparse_LIBRARY})

    ADD_LIBRARY(cupti SHARED IMPORTED GLOBAL)
    SET_PROPERTY(TARGET cupti PROPERTY IMPORTED_LOCATION ${CUDA_cupti_LIBRARY})

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

endif(WITH_GPU)
