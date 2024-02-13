cmake_minimum_required(VERSION 3.12)
project(Main)
SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)

file(GLOB SURF_SRC *.cpp *.h)

add_executable(SURF ${SURF_SRC})

FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(SURF ${OpenCV_LIBS})
ELSE(OpenCV_FOUND)
    MESSAGE("OpenCV not Found!")
ENDIF(OpenCV_FOUND)

file(GLOB IMG images/*)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/images)
foreach(img ${IMG})
    configure_file(${img} images/ COPYONLY)
endforeach()