cmake_minimum_required(VERSION 3.14)

# ============================================================================
# rowmap_navigation — 行结构地图视觉导航核心 (独立工程)
#
# 编译产物模块名 vision_navigation (接口兼容板端既有导入路径),
# 内部实现为 RowMap 算法, 与部署目标板端工程相互独立。
#
# PC 构建 (Windows, VS 生成器):
#   cmake -S rowmap_navigation -B rowmap_navigation/build-pc ^
#       -G "Visual Studio 18 2026" -A x64 ^
#       -DPython_EXECUTABLE="<venv>/Scripts/python.exe" ^
#       -Dpybind11_DIR="<venv>/Lib/site-packages/pybind11/share/cmake/pybind11"
#   cmake --build rowmap_navigation/build-pc --config Release
#
# RK3588 板端构建 (aarch64, 板上原生):
#   cmake -S rowmap_navigation -B rowmap_navigation/build-rk3588 \
#       -DPython_EXECUTABLE=$(which python3) \
#       -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
#   cmake --build rowmap_navigation/build-rk3588 -j4
# ============================================================================

project(rowmap_navigation_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Python 与 pybind11 ──
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 优先使用 venv 内 pip 安装的 pybind11 (通过 -Dpybind11_DIR 传入),
# 回退到系统安装
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(FATAL_ERROR
        "pybind11 not found. 传入 -Dpybind11_DIR=<site-packages>/pybind11/share/cmake/pybind11 "
        "或在目标机安装 pybind11-dev")
endif()

# ── 模块: vision_navigation ──
pybind11_add_module(vision_navigation
    src/bindings.cpp
)

target_include_directories(vision_navigation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# MSVC: 源码为 UTF-8 (含中文注释), 显式声明字符集避免 C4819 告警;
# NOMINMAX 防止 windows.h min/max 宏污染 (板端 GCC/Clang 无副作用)
target_compile_definitions(vision_navigation PRIVATE NOMINMAX)
target_compile_options(vision_navigation PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)

set_target_properties(vision_navigation PROPERTIES
    PREFIX ""
    OUTPUT_NAME "vision_navigation"
)

# 编译产物同步到 lib/ (部署目录: 拷贝 .so/.pyd 到板端工程 navigation/ 即完成替换)
add_custom_command(TARGET vision_navigation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:vision_navigation>
        ${CMAKE_SOURCE_DIR}/lib/$<TARGET_FILE_NAME:vision_navigation>
    COMMENT "Syncing vision_navigation module -> rowmap_navigation/lib/"
)
