cmake_minimum_required(VERSION 3.15)
project(Main)
SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)

include_directories(./Eigen)
add_executable(Main main.cpp)


FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(Main ${OpenCV_LIBS})
ELSE(OpenCV_FOUND)
    MESSAGE("OpenCV not Found!")
ENDIF(OpenCV_FOUND)