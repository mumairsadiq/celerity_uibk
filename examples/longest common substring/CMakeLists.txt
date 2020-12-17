cmake_minimum_required(VERSION 3.5.1)
project(LCSubstr)

find_package(Celerity CONFIG REQUIRED)
add_executable(LCSubstr LCSubstr.cpp)
add_celerity_to_target(TARGET LCSubstr SOURCES LCSubstr.cpp)