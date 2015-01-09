cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ../src )

add_executable( frestest ../src/FileReader.cc ../src/Debug.cc ../src/Array.cc ../src/TextRecognition.cc ../src/Dictionary.cc ../tests/frestest.cc )
add_executable( dicttest ../src/FileReader.cc ../src/Debug.cc ../src/Array.cc ../src/TextRecognition.cc ../src/Dictionary.cc ../tests/dicttest.cc )
add_executable( debug ../src/FileReader.cc ../src/Debug.cc ../src/Array.cc ../src/TextRecognition.cc ../src/Dictionary.cc ../tests/debug.cc )

target_link_libraries( frestest ${OpenCV_LIBS} )
target_link_libraries( dicttest ${OpenCV_LIBS} )
target_link_libraries( debug ${OpenCV_LIBS} )