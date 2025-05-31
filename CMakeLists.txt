# Minimum CMake version required
cmake_minimum_required(VERSION 3.15) # Using 3.15 for good generator expression support

# Project Name and Language
project(MyCppProgram CXX) # You can rename "MyCppProgram"

# Executable name
set(EXECUTABLE_NAME test)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Good practice to disable GCC-specific extensions for portability

 set(SRC
     src/baseline.cpp
     src/io.cpp
 )

add_executable(${EXECUTABLE_NAME} ${SRC})

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_compile_options(${EXECUTABLE_NAME} PRIVATE
    -Wall
    -Wextra
)

# Debug builds.
target_compile_options(${EXECUTABLE_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g>          # Add debug symbols
)
target_compile_definitions(${EXECUTABLE_NAME} PRIVATE
    $<$<CONFIG:Debug>:DEBUG>       # Define DEBUG macro (e.g., for #ifdef DEBUG)
)

# Release builds:
target_compile_options(${EXECUTABLE_NAME} PRIVATE
    $<$<CONFIG:Release>:-O3>        # High optimization
)
target_compile_definitions(${EXECUTABLE_NAME} PRIVATE
    $<$<CONFIG:Release>:NDEBUG>     # Define NDEBUG macro (disables assert(), etc.)
)

# Print a message upon configuration
message(STATUS "Configuring ${PROJECT_NAME} for target ${EXECUTABLE_NAME}")
message(STATUS "Source files: ${SRC}")
message(STATUS "Include directory: ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "Build types available (typically): Debug, Release, RelWithDebInfo, MinSizeRel")