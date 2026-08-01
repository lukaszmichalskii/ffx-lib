cmake_minimum_required(VERSION 3.25.2)
project(ffx VERSION 1.0.0 LANGUAGES CXX)

option(FFX_BUILD_TESTING "Build the ffx unit tests" ON)
option(FFX_BUILD_CODEGEN_TESTING "Build the ffx codegen for testing emitted code" OFF)

option(FFX_ALGORITHM_ENABLED "Build the ffx::algorithm extension" ON)
option(FFX_FRAMEWORK_ENABLED "Build the ffx::framework extension" ON)
option(FFX_NN_ENABLED "Build the ffx::nn extension" ON)

if (NOT FFX_NN_ENABLED)
    set(FFX_BUILD_CODEGEN_TESTING OFF)
endif ()

include(FetchContent)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

find_package(Boost REQUIRED COMPONENTS atomic)
find_package(alpaka 2.1.1 QUIET)

if(NOT alpaka_FOUND)
    FetchContent_Declare(
        alpaka
        URL https://github.com/alpaka-group/alpaka/archive/refs/tags/2.1.1.tar.gz
        SYSTEM
    )
    FetchContent_MakeAvailable(alpaka)
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/ffx/version.h"
    @ONLY
)

add_library(ffx INTERFACE)
add_library(ffx::ffx ALIAS ffx)

target_link_libraries(ffx INTERFACE
    $<BUILD_INTERFACE:alpaka::alpaka>
    $<BUILD_INTERFACE:Boost::atomic>
)

target_include_directories(ffx INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(FFX_ALGORITHM_ENABLED)
    target_compile_definitions(ffx INTERFACE FFX_ALGORITHM_ENABLED)
endif()

if(FFX_FRAMEWORK_ENABLED)
    target_compile_definitions(ffx INTERFACE FFX_FRAMEWORK_ENABLED)
endif()

if(FFX_NN_ENABLED)
    target_compile_definitions(ffx INTERFACE FFX_NN_ENABLED)
endif()

if(FFX_BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

if(FFX_BUILD_CODEGEN_TESTING AND FFX_NN_ENABLED)
    add_subdirectory(codegen_test)
endif ()

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ffxConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ffxConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ffxConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ffx
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ffx EXPORT ffxTargets)

install(
    EXPORT ffxTargets
    FILE ffxTargets.cmake
    NAMESPACE ffx::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ffx
)

install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ffxConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ffxConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ffx
)