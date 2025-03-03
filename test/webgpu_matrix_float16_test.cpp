#include <gtest/gtest.h>
#include <stdfloat>

import cpp_matrix;

using Matrix = cpp_matrix::WebGpuMatrix<std::float16_t>;

#define MATRIX_TEST_NAME WebGpuMatrixFloat16Test

#include "matrix_test.cpp"