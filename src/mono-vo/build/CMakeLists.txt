cmake_minimum_required(VERSION 2.8)
project(mono-vo)

find_package(OpenCV 3.0 REQUIRED)
include_directories(include "/usr/include/eigen3")

message(STATUS ${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_LIBS})

file(GLOB viso
    "../src/*.h"
    "../src/*.cpp"
)

#add_executable(mono_vo_ros src/visodo_ros.cpp )
add_executable(vo ../src/visodo.cpp)


target_link_libraries(vo ${OpenCV_LIBS} )
#target_link_libraries( mono_vo_ros
#  ${OpenCV_LIBS}
#)
