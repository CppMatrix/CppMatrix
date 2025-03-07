#include <gtest/gtest.h>
#include <stdfloat>

import cpp_matrix;

using Matrix = cpp_matrix::CpuMatrix<std::float16_t>;

#define MATRIX_TEST_NAME CpuMatrixFloat16Test

#include "matrix_test.cpp"
#include "neural_network_test.cpp"