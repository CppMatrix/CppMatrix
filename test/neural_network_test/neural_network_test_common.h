#pragma once

#ifndef MATRIX_TEST_NAME
#error MATRIX_TEST_NAME must be defined.
#endif

#include <gtest/gtest.h>

#define XSTR(x) STR(x)
#define STR(x) #x
#define CONN2(X, Y) X##Y
#define CONN3(X, Y, Z) X##Y##Z
#define CONN(X, Y) CONN2(X, Y)
#define NEURAL_NETWORK_TEST_NAME CONN(MATRIX_TEST_NAME, _NeuralNetworkTest)

class NEURAL_NETWORK_TEST_NAME : public testing::Test {
public:
    void SetUp() override
    {
        if (!Matrix::IsAvaliable()) {
            GTEST_SKIP() << XSTR(MATRIX_TEST_NAME) << " is not avaliable, skip.";
        }
    }
};

#define NEURAL_NETWORK_TEST(X, Y) TEST_F(NEURAL_NETWORK_TEST_NAME, CONN3(X, _, Y))

using namespace cpp_matrix;
using namespace cpp_matrix::neural_network;

using Tensor = cpp_matrix::neural_network::Tensor<Matrix>;