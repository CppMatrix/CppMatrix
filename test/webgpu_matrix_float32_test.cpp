#include <gtest/gtest.h>
#include <stdfloat>

import cpp_matrix;

using Matrix = cpp_matrix::experiment::WebGpuMatrix<std::float32_t>;

#define MATRIX_TEST_NAME DISABLED_WebGpuMatrixFloat32Test

#include "matrix_test.cpp"