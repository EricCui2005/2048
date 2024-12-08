cmake_minimum_required(VERSION 3.5)
project(Learning2048 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define all source files
set(SOURCE_FILES
    src/main.cpp
    src/game.cpp
    src/mcts.cpp
    src/gameRunner.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Add include directory
target_include_directories(${PROJECT_NAME} 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create data directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)

# Set output directories
set_target_properties(${PROJECT_NAME} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)