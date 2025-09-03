cmake_minimum_required(VERSION 3.5)
project(plate_extractor)

find_package(OpenCV REQUIRED)

add_executable(plate_extractor main.cpp)
target_link_libraries(plate_extractor PRIVATE ${OpenCV_LIBS})
