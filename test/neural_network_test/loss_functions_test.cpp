#ifndef MATRIX_TEST_NAME
#error MATRIX_TEST_NAME must be defined.
#endif

#include <gtest/gtest.h>

#define XSTR(x) STR(x)
#define STR(x) #x
#define CONN2(X, Y) X##Y
#define CONN(X, Y) CONN2(X, Y)
#define NEURAL_NETWORK_TEST_NAME CONN(MATRIX_TEST_NAME, _NeuralNetwork_LossFunctionTest)

class NEURAL_NETWORK_TEST_NAME : public testing::Test {
public:
    void SetUp() override
    {
        if (!Matrix::IsAvaliable()) {
            GTEST_SKIP() << XSTR(MATRIX_TEST_NAME) << " is not avaliable, skip.";
        }
    }
};

#define NEURAL_NETWORK_TEST(X) TEST_F(NEURAL_NETWORK_TEST_NAME, X)

import cpp_matrix.neural_network;

using namespace cpp_matrix;
using namespace cpp_matrix::neural_network;

NEURAL_NETWORK_TEST(MeanSquaredErrorLoss)
{
    Matrix a {
        { 1.0_mf, 1.1_mf, 1.2_mf },
        { 2.0_mf, 2.2_mf, 2.3_mf },
    };

    Matrix b {
        { 1.01_mf, 1.11_mf, 1.12_mf },
        { 2.10_mf, 2.11_mf, 2.21_mf },
    };

    auto c = MeanSquaredErrorLoss(a, b);
    ASSERT_EQ(c.Row(), 1);
    ASSERT_EQ(c.Column(), 1);

    Matrix::ElementType sum {};
    sum += (1.0_mf - 1.01_mf) * (1.0_mf - 1.01_mf);
    sum += (1.1_mf - 1.11_mf) * (1.1_mf - 1.11_mf);
    sum += (1.2_mf - 1.12_mf) * (1.2_mf - 1.12_mf);
    sum += (2.0_mf - 2.10_mf) * (2.0_mf - 2.10_mf);
    sum += (2.2_mf - 2.11_mf) * (2.2_mf - 2.11_mf);
    sum += (2.3_mf - 2.21_mf) * (2.3_mf - 2.21_mf);
    ASSERT_FLOAT_EQ(c.Read()[0], sum / 6);
}