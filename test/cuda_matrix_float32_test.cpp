#include <gtest/gtest.h>
#include <stdfloat>

import cpp_matrix;

using Matrix = cpp_matrix::CudaMatrix<std::float32_t>;

#define MATRIX_TEST_NAME CudaMatrixFloat32Test

#include "matrix_test.cpp"