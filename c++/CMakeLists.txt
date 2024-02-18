cmake_minimum_required(VERSION 3.16)
project(main_tf_lite C CXX)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project" )
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    # "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
    "${CMAKE_CURRENT_LIST_DIR}/tensorflow_src" ABSOLUTE)
endif()

SET("OpenCV_DIR" "/usr/share/OpenCV")
find_package( OpenCV REQUIRED )

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(main_tflite_model ${CMAKE_CURRENT_LIST_DIR}/main_tflite_model.cpp)
target_link_libraries(main_tflite_model tensorflow-lite jsoncpp ${OpenCV_LIBS} OpenCL)