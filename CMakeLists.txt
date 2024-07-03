cmake_minimum_required(VERSION 3.10)
project(transformer)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA paths
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# CUDA includes and link libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

# Find the libtorch package
find_package(Torch REQUIRED)

# Find Python and NumPy
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

# Include matplotlib-cpp
include_directories(/opt/matplotlib-cpp/matplotlib-cpp)


# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add executable
add_executable(transformer ${SOURCES})

# Link libraries
target_link_libraries(transformer ${CUDA_LIBRARIES} "${TORCH_LIBRARIES}" ${Python3_LIBRARIES} ${Python3_NumPy_LIBRARIES})
