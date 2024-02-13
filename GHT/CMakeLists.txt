cmake_minimum_required(VERSION 3.12)
project(Main)

set(CMAKE_CXX_STANDARD 14)

add_executable(Main main.cpp)


FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(Main ${OpenCV_LIBS})
ELSE(OpenCV_FOUND)
    MESSAGE("OpenCV not Found!")
ENDIF(OpenCV_FOUND)