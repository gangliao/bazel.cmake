<div align="center">
    <img src="https://raw.githubusercontent.com/CMakeHub/bazaar/master/logo-name.png" width="20%"><br><br>
</div>

-----------------

<center>

| **`Linux CPU|GPU`** | **`Mac OS CPU|GPU`** | **`Windows CPU`** | **`Android`** | **`Apple IOS`** |
|---------------------|----------------------|-------------------|---------------|-----------------|
| [![Build Status](https://travis-ci.com/CMakeHub/demo.bazel.cmake.svg?branch=master)](https://travis-ci.com/CMakeHub/demo.bazel.cmake)            |  [![Build Status](https://travis-ci.com/CMakeHub/demo.bazel.cmake.svg?branch=master)](https://travis-ci.com/CMakeHub/demo.bazel.cmake)                   | [![Build Status](https://travis-ci.com/CMakeHub/demo.bazel.cmake.svg?branch=master)](https://travis-ci.com/CMakeHub/demo.bazel.cmake)                 |   [![Build status](https://ci.appveyor.com/api/projects/status/2leddlgpdfsmqmca?svg=true)](https://ci.appveyor.com/project/gangliao/demo-bazel-cmake)       |     [![Build Status](https://travis-ci.com/CMakeHub/demo.bazel.cmake.svg?branch=master)](https://travis-ci.com/CMakeHub/demo.bazel.cmake)          |

</center>

**bazel.cmake** is a seamless submodule which aims to mimic the behavior of bazel to simplify the usability of CMake for **any mainstream operating systems** (Linux, Mac OS, Windows, Android, IOS). 

When launching a large open source project, We found that not everyone can adapt to the tremendous higher-level mechanisms CMake offered. We borrow the abstracts from [Bazel](https://bazel.build/) to make it easy for us to do so! Back to 2017, Bazel has not yet matured to support manycore accelerators and even the most popular operating system - Windows. Maybe it's better now, but CMake is still the most powerful building tool for C++.  Last but not least, developers can leverage the elegant abstracts from this module to reach the fine-grained compilation and testing. So we will continue to support this project.

## Preparation

If you create a new project, first add git submodule `bazel.cmake` into your project.

```bash
project_dir$ git submodule add --force https://github.com/CMakeHub/bazel.cmake
project_dir$ git submodule update --init --recursive
```

Just like our demo's repository [**demo.bazel.cmake**](https://github.com/CMakeHub/demo.bazel.cmake), you need to integrate `bazel.cmake` module into 
current [project's CMakeLists.txt](https://github.com/CMakeHub/demo.bazel.cmake/blob/b6d882c706e4d0ea16cf2152489af9b583b94537/CMakeLists.txt#L23-L26) as follows:

```cmake
# CMakeLists.txt
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/bazel.cmake/cmake)
include(bazel)
```

Then, you can use the built-in **bazel abstracts** to compile your code and run it under any mainstream operating system!

## Compile Your Code

To compile the [following code](https://github.com/CMakeHub/demo.bazel.cmake/blob/master/c%2B%2B/hello.cc), you can invoke `cc_library` in [CMakeLists.txt](https://github.com/CMakeHub/demo.bazel.cmake/blob/0cd58ddaf4e004f4008363a1c4dfefac457bc279/c%2B%2B/CMakeLists.txt#L4).

```c++
// hello.cc
#include <stdio.h>

int main(int argc, char *argv[]) {
    printf("Hello World!\n");
    return 0;
}
```

```cmake
# CMakeLists.txt
cc_binary(hello SRCS hello.cc)
```

Then, issue the below commands to build an executable for **Mac OS or Linux**.

```bash
project_dir$ mkdir build && cd build
project_dir$ cmake ..
project_dir$ make hello
```

```bash
# Build Output
/build> make hello
Scanning dependencies of target hello
[ 50%] Building CXX object c++/CMakeFiles/hello.dir/hello.cc.o
[100%] Linking CXX executable hello
[100%] Built target hello
```

## Advanced Options

### Build on Windows

Your host machine must be Windows!

```bash
mkdir build && cd build
cmake .. -G "Visual Studio 14 2015" -DCMAKE_GENERATOR_PLATFORM=x64 -Wno-dev
msbuild testcase.sln /p:BuildInParallel=true /m:4
```

### Build on Android

```bash
# Download and decompress Android ndk 
wget -c https://dl.google.com/android/repository/android-ndk-r14b-darwin-x86_64.zip && unzip -q android-ndk-r14b-darwin-x86_64.zip
ANDROID_STANDALONE_TOOLCHAIN=`pwd`/android-toolchain-gcc
android-ndk-r14b/build/tools/make-standalone-toolchain.sh --force --arch=arm --platform=android-21 --install-dir=$ANDROID_STANDALONE_TOOLCHAIN

# Create the build directory for CMake.
mkdir build && cd build
PROJECT_DIR=...
cmake -DCMAKE_TOOLCHAIN_FILE=$PROJECT_DIR/bazel.cmake/third-party/android-cmake/android.toolchain.cmake \
      -DANDROID_STANDALONE_TOOLCHAIN=$ANDROID_STANDALONE_TOOLCHAIN \
      -DANDROID_ABI="armeabi-v7a with NEON FP16" \
      -DANDROID_NATIVE_API_LEVEL=21 \
      ..
make -j4
```

### Build on Apple IOS

```bash
mkdir build && cd build
PROJECT_DIR=...
cmake -DCMAKE_TOOLCHAIN_FILE=$PROJECT_DIR/bazel.cmake/third-party/ios-cmake/toolchain/iOS.cmake \
      -DIOS_PLATFORM=OS \
      ..
make "-j$(sysctl -n hw.ncpu)" VERBOSE=1
```

## Cheat Sheet

```bash
| APIs                                                | Linux | Windows | Android | Mac OS X | Apple IOS |
|-----------------------------------------------------|-------|---------|---------|----------|-----------|
| cc_library(lib_name SRCS src1.cc... [DEPS lib1...]) |  yes  |   yes   |   yes   |    yes   |    yes    |
| cc_library(lib_name DEPS lib1...)                   |  yes  |   TBD   |   yes   |    yes   |    yes    |
| cc_testing(bin_name SRCS src1.cc... [DEPS lib1...]) |  yes  |   yes   |   yes   |    yes   |    yes    |
| cc_binary(bin_name SRCS src1.cc... [DEPS lib1...])  |  yes  |   yes   |   yes   |    yes   |    yes    |
| nv_library(lib_name SRCS src1.cc... [DEPS lib1...]) |  yes  |   yes   | no cuda |    yes   |  no cuda  |
| nv_library(lib_name DEPS lib1...)                   |  yes  |   TBD   | no cuda |    yes   |  no cuda  |
| nv_testing(bin_name SRCS src1.cc... [DEPS lib1...]) |  yes  |   yes   | no cuda |    yes   |  no cuda  |
| nv_binary(bin_name SRCS src1.cc... [DEPS lib1...])  |  yes  |   yes   | no cuda |    yes   |  no cuda  |

Note: [DEPS lib1...] is optional syntax rules.

# To build a static library example.a from example.cc using the system
#  compiler (like GCC):
#
    cc_library(example SRCS example.cc)
#
# To build a static library example.a from multiple source files
# example{1,2,3}.cc:
#
    cc_library(example SRCS example1.cc example2.cc example3.cc)
#
# To build a shared library example.so from example.cc:
#
    cc_library(example SHARED SRCS example.cc)
#
# To build a library using Nvidia's NVCC from .cu file(s), use the nv_
# prefixed version:
#
    nv_library(example SRCS example.cu)
#
# To specify that a library new_example.a depends on other libraies:
#
    cc_library(new_example SRCS new_example.cc DEPS example)
#
# Static libraries can be composed of other static libraries:
#
    cc_library(composed DEPS dependent1 dependent2 dependent3)
#
# To build an executable binary file from some source files and
# dependent libraries:
#
    cc_binary(example SRCS main.cc something.cc DEPS example1 example2)
#
# To build an executable binary file using NVCC, use the nv_ prefixed
# version:
#
    nv_binary(example SRCS main.cc something.cu DEPS example1 example2)
#
# To build a unit test binary, which is an executable binary with
# GoogleTest linked:
#
    cc_testing(example_test SRCS example_test.cc DEPS example)
#
# To build a unit test binary using NVCC, use the nv_ prefixed version:
#
    nv_testing(example_test SRCS example_test.cu DEPS example)
#
# It is pretty often that executable and test binaries depend on
# pre-defined external libaries like glog and gflags defined in
# /cmake/external/*.cmake:
#
    cc_testing(example_test SRCS example_test.cc DEPS example glog gflags)
#
```

## TODO

- [ ] Add proto_library

## License

[Apache License 2.0](LICENSE)
