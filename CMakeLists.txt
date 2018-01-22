cmake_minimum_required(VERSION 3.8)
project(Computer_Vision)
find_package(OpenCV3)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp adaptiveThreshold.h boxFilter.h GLCM.h histEqual.h gaussFilter.h alphaChannel.h alphaChannel.h)
add_executable(Computer_Vision ${SOURCE_FILES})
find_package(OpenCV REQUIRED)
target_link_libraries(Computer_Vision ${OpenCV_LIBS})