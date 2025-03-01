#include <gtest/gtest.h>

import cpp_matrix;

#define MATRIX_TEST(X) TEST(CudaMatrixFloat16Test, X)

using Matrix = cpp_matrix::CudaMatrix<std::float16_t>;

#include "matrix_test.cpp"