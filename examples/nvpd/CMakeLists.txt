cmake_minimum_required(VERSION 3.5.1)
project(nvpd)

find_package(Celerity CONFIG REQUIRED)
add_executable(nvpd nvpd.cpp)
add_celerity_to_target(TARGET nvpd SOURCES nvpd.cpp)
