#include <gtest/gtest.h>

import cpp_matrix;

#define MATRIX_TEST(X) TEST(CudaMatrixFloat32Test, X)

using Matrix = cpp_matrix::CudaMatrix<std::float32_t>;

#include "matrix_test.cpp"