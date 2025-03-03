#include <gtest/gtest.h>
#include <stdfloat>

import cpp_matrix;

using Matrix = cpp_matrix::experiment::WebGpuMatrix<std::float16_t>;

#define MATRIX_TEST_NAME DISABLED_WebGpuMatrixFloat16Test

#include "matrix_test.cpp"