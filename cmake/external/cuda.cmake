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
endif(WITH_GPU)
