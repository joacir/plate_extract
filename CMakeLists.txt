cmake_minimum_required(VERSION 3.5)
project(plate_extractor)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED tesseract)
pkg_check_modules(LEPT REQUIRED lept)

add_executable(plate_extractor main.cpp)
target_compile_features(plate_extractor PRIVATE cxx_std_17)
target_include_directories(plate_extractor PRIVATE ${OpenCV_INCLUDE_DIRS} ${TESSERACT_INCLUDE_DIRS} ${LEPT_INCLUDE_DIRS})
target_link_libraries(plate_extractor PRIVATE ${OpenCV_LIBS} ${TESSERACT_LIBRARIES} ${LEPT_LIBRARIES})
