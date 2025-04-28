cmake_minimum_required(VERSION 3.14)
project(model-qa VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)

# Print system information
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find ONNX Runtime using CMake config
find_package(onnxruntime REQUIRED)
if(NOT onnxruntime_FOUND)
    message(FATAL_ERROR "ONNX Runtime not found. Please install it with:\n"
                        "  sudo apt-get install libonnxruntime-dev\n"
                        "  or build from source: https://github.com/microsoft/onnxruntime")
endif()

# Find nlohmann_json
find_package(nlohmann_json REQUIRED)
if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json not found. Please install it with:\n"
                        "  sudo apt-get install nlohmann-json3-dev")
endif()

# Download cxxopts if not present
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cxxopts.hpp")
    message(STATUS "Downloading cxxopts.hpp...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/jarro2783/cxxopts/master/include/cxxopts.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/cxxopts.hpp"
        SHOW_PROGRESS
    )
endif()

# Check for tokenizers libraries
set(USE_TOKENIZERS_CPP OFF)
if(EXISTS "/usr/local/lib/libtokenizers_cpp.a" AND EXISTS "/usr/local/lib/libtokenizers_c.a")
    set(USE_TOKENIZERS_CPP ON)
    set(TOKENIZERS_CPP_LIBRARY "/usr/local/lib/libtokenizers_cpp.a")
    set(TOKENIZERS_C_LIBRARY "/usr/local/lib/libtokenizers_c.a")
    message(STATUS "Found tokenizers-cpp static libraries")
endif()

# Set compiler flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Add executable
add_executable(model-qa model-qa.cpp)

# Include directories
target_include_directories(model-qa PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # For cxxopts.hpp
    ${nlohmann_json_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(model-qa PRIVATE
    onnxruntime::onnxruntime
    nlohmann_json::nlohmann_json
)

# Add tokenizers if available
if(USE_TOKENIZERS_CPP)
    target_link_libraries(model-qa PRIVATE 
        ${TOKENIZERS_CPP_LIBRARY}
        ${TOKENIZERS_C_LIBRARY}
    )
    target_compile_definitions(model-qa PRIVATE USE_TOKENIZERS_CPP)
    target_include_directories(model-qa PRIVATE /usr/local/include)
endif()

# Install target
install(TARGETS model-qa
    RUNTIME DESTINATION bin
) 