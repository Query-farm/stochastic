cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME quack)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(boost_math CONFIG REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

set(CMAKE_CXX_STANDARD 20)

project(${TARGET_NAME})
include_directories(src/include)

set(EXTENSION_SOURCES src/quack_extension.cpp
src/distribution_normal.cpp
src/distribution_beta.cpp
src/distribution_bernoulli.cpp
src/distribution_laplace.cpp
src/distribution_lognormal.cpp
src/distribution_cauchy.cpp
src/distribution_gamma.cpp
src/distribution_geometric.cpp
src/distribution_pareto.cpp
src/distribution_uniform_real.cpp
src/distribution_uniform_int.cpp
src/rng_utils.cpp
)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link OpenSSL in both the static library as the loadable extension
target_link_libraries(${EXTENSION_NAME} Boost::math)
target_link_libraries(${LOADABLE_EXTENSION_NAME} Boost::math)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
