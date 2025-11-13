cmake_minimum_required(VERSION 4.0)
project(111)

set(CMAKE_CXX_STANDARD 20)

add_executable(111 main.cpp)

set(LIBTORCH_PATH /Users/chasen/Desktop/111/libtorch)
target_include_directories(${PROJECT_NAME} PRIVATE
        ${LIBTORCH_PATH}/include/torch/csrc/api/include
        ${LIBTORCH_PATH}/include)  # 如果有单独的 API include 目录

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${LIBTORCH_PATH}/lib/libtorch.dylib
        ${LIBTORCH_PATH}/lib/libtorch_cpu.dylib
        ${LIBTORCH_PATH}/lib/libomp.dylib
        ${LIBTORCH_PATH}/lib/libc10.dylib
        ${LIBTORCH_PATH}/lib/libshm.dylib
        ${LIBTORCH_PATH}/lib/libtorch_global_deps.dylib)

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/resources"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/resources"
        COMMENT "Copying resources...")
