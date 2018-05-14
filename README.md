<div align="center">
    <img src="https://raw.githubusercontent.com/CMakeHub/bazaar/master/logo-name.png" width="20%"><br><br>
</div>

-----------------

<center>

| **`Linux CPU`** | **`Linux GPU`** | **`Mac OS CPU`** | **`Windows CPU`** | **`Android`** | **`Apple IOS`** |
|-----------------|---------------------|------------------|-------------------|---------------|---------------|
| [![Build Status](https://travis-ci.com/CMakeHub/demo.bazel.cmake.svg?branch=master)](https://travis-ci.com/CMakeHub/demo.bazel.cmake)            |  [![Build Status](https://travis-ci.com/CMakeHub/demo.bazel.cmake.svg?branch=master)](https://travis-ci.com/CMakeHub/demo.bazel.cmake)                   | [![Build Status](https://travis-ci.com/CMakeHub/demo.bazel.cmake.svg?branch=master)](https://travis-ci.com/CMakeHub/demo.bazel.cmake)                 |   [![Build status](https://ci.appveyor.com/api/projects/status/2leddlgpdfsmqmca?svg=true)](https://ci.appveyor.com/project/gangliao/demo-bazel-cmake)       |     [![Build Status](https://travis-ci.com/CMakeHub/demo.bazel.cmake.svg?branch=master)](https://travis-ci.com/CMakeHub/demo.bazel.cmake)          |           [![Build Status](https://travis-ci.com/CMakeHub/demo.bazel.cmake.svg?branch=master)](https://travis-ci.com/CMakeHub/demo.bazel.cmake)          |

</center>

**bazel.cmake** is a seamless submodule which aims to mimic the behavior of bazel to simplify the usability of CMake for **any mainstream operating systems** (Linux, Mac OS, Windows, Android, IOS). 

When launching a large open source project, We found that not everyone can adapt to the tremendous higher-level mechanisms CMake offered. We borrow the abstracts from [Bazel](https://bazel.build/) to make it easy for us to do so! Back to 2017, Bazel has not yet matured to support the manycore accelerators and even the most popular operating system - Windows. Maybe it's better now, but CMake is still the most powerful building tool for C++. So we will continue to support this project.

## How to use it ?

If you create a new project, first add git submodule `bazel.cmake` into your project.

```bash
project_dir$ git submodule add --force https://github.com/CMakeHub/bazel.cmake
project_dir$ git submodule update --init --recursive
```

Then, just like our demo project [demo.bazel.cmake](https://github.com/CMakeHub/demo.bazel.cmake), you need to integrate `bazel.cmake` module into 
current [project's CMakeLists.txt](https://github.com/CMakeHub/demo.bazel.cmake/blob/b6d882c706e4d0ea16cf2152489af9b583b94537/CMakeLists.txt#L23-L26) as follows:

```cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/bazel.cmake/cmake)
include(bazel)
```



## License

[Apache License 2.0](LICENSE)
