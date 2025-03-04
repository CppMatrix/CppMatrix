#include <gtest/gtest.h>
#include <stdfloat>

import porch;

using Tensor = porch::Tensor<BACKEND>;

#define CONN(X, Y) X##Y
#define TENSOR_TEST(X, Y, Z) TEST(CONN(X, Y), Z)

TENSOR_TEST(TensorTest, BACKEND_NAME, Initializing)
{
    Tensor data {};
}
