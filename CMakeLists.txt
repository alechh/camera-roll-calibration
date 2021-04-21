cmake_minimum_required(VERSION 3.17)
project(test_opencv)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_opencv main.cpp SpaceKB.cpp SpaceKB.h Interval.cpp Interval.h IntervalsList.cpp IntervalsList.h ListOfIntervalsLists.cpp ListOfIntervalsLists.h)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( test_opencv ${OpenCV_LIBS} )
