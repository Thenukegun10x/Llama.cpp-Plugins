cmake_minimum_required(VERSION 3.22)
project(turbo-quant-plugin LANGUAGES CXX)

add_library(turbo-quant-plugin SHARED
    turbo_quant_plugin.cpp
    turboquant.cpp
)

target_include_directories(turbo-quant-plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../SageAttention-rocm/plugin-system
)

set_target_properties(turbo-quant-plugin PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    LINKER_LANGUAGE CXX
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

if(MSVC)
    target_compile_options(turbo-quant-plugin PRIVATE /W3 /utf-8)
else()
    target_compile_options(turbo-quant-plugin PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS turbo-quant-plugin
    LIBRARY DESTINATION plugins/
    RUNTIME DESTINATION plugins/
)
