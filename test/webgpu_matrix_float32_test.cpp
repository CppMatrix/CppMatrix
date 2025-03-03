#include <gtest/gtest.h>
#include <stdfloat>

import cpp_matrix;

using Matrix = cpp_matrix::WebGpuMatrix<std::float32_t>;

#define MATRIX_TEST_NAME WebGpuMatrixFloat32Test

#include "matrix_test.cpp"