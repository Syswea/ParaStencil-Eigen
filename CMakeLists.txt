cmake_minimum_required(VERSION 3.18)
project(SoftmaxMPI LANGUAGES CXX)

# --- 设置 Clang 编译器 ---
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")

# --- C++ 标准 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- 查找 MPI ---
find_package(MPI REQUIRED COMPONENTS CXX)

# --- 修复 Eigen 路径（针对 apt 安装的 libeigen3-dev）---
# Eigen 头文件实际位于 /usr/include/eigen3/Eigen/
include_directories(/usr/include/eigen3)

# --- 添加可执行文件 ---
add_executable(softmax softmax.cpp)

# --- 链接 MPI ---
target_link_libraries(softmax PRIVATE MPI::MPI_CXX)

# --- 编译选项 ---
target_compile_options(softmax PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
)

# --- 自定义 'run' 目标 ---
find_program(MPIRUN mpirun REQUIRED)
if(NOT DEFINED RUN_N)
    set(RUN_N 4)
endif()

add_custom_target(run
    COMMAND ${MPIRUN} -n ${RUN_N} $<TARGET_FILE:softmax>
    DEPENDS softmax
    COMMENT "Running softmax with ${RUN_N} MPI processes..."
)