cmake_minimum_required(VERSION 2.8.9)
project(directory_test)
set(CMAKE_BUILD_TYPE Release)

include_directories(object_tracking)

file(GLOB SOURCES "*.cc")
file(GLOB SOURCES "object_tracking/*.cc")

add_library(elib SHARED ${SOURCES})
