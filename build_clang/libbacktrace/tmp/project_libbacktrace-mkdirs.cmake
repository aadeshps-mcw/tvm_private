# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/mcw/aadesh/tvm/cmake/libs/../../3rdparty/libbacktrace"
  "/home/mcw/aadesh/tvm/build_clang/libbacktrace"
  "/home/mcw/aadesh/tvm/build_clang/libbacktrace"
  "/home/mcw/aadesh/tvm/build_clang/libbacktrace/tmp"
  "/home/mcw/aadesh/tvm/build_clang/libbacktrace/src/project_libbacktrace-stamp"
  "/home/mcw/aadesh/tvm/build_clang/libbacktrace/src"
  "/home/mcw/aadesh/tvm/build_clang/libbacktrace/src/project_libbacktrace-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/mcw/aadesh/tvm/build_clang/libbacktrace/src/project_libbacktrace-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/mcw/aadesh/tvm/build_clang/libbacktrace/src/project_libbacktrace-stamp${cfgdir}") # cfgdir has leading slash
endif()
