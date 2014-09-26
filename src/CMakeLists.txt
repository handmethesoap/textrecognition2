cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( . )

add_executable( textrecognition FileReader.cc Debug.cc Array.cc TextRecognition.cc Dictionary.cc main.cc )

target_link_libraries( textrecognition ${OpenCV_LIBS} )
