cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ../src )

add_executable( frestest ../src/FileReader.cc ../src/Debug.cc ../src/Array.cc ../src/TextRecognition.cc ../src/Dictionary.cc ../src/main.cc )

target_link_libraries( frestest ${OpenCV_LIBS} )
