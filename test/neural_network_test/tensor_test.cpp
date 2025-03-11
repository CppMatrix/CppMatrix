#ifndef MATRIX_TEST_NAME
#error MATRIX_TEST_NAME must be defined.
#endif

#include <gtest/gtest.h>

#define XSTR(x) STR(x)
#define STR(x) #x
#define CONN2(X, Y) X##Y
#define CONN(X, Y) CONN2(X, Y)
#define TENSOR_TEST_NAME CONN(MATRIX_TEST_NAME, _NeuralNetwork_TensorTest)

class TENSOR_TEST_NAME : public testing::Test {
public:
    void SetUp() override
    {
        if (!Matrix::IsAvaliable()) {
            GTEST_SKIP() << XSTR(MATRIX_TEST_NAME) << " is not avaliable, skip.";
        }
    }
};

#define TENSOR_TEST(X) TEST_F(TENSOR_TEST_NAME, X)

import cpp_matrix.neural_network;

using namespace cpp_matrix;
using Tensor = cpp_matrix::neural_network::Tensor<Matrix>;

TENSOR_TEST(AutoGrad)
{
    auto x = Tensor { "x",
        {
            { 1.0_mf, 1.1_mf, 1.2_mf },
            { 2.0_mf, 2.2_mf, 2.3_mf },
        } };

    auto y = Tensor { "y",
        {
            { 1.0_mf, 1.1_mf, 1.2_mf },
            { 2.0_mf, 2.2_mf, 2.3_mf },
        } };

    auto z = 3.0_mf * x.Pow(2) + 2.0_mf * y;
    auto dx = z.Backward("x");
    auto dy = z.Backward("y");

    auto expectDx
        = [](typename Matrix::ElementType v) { return 3.0_mf * 2.0_mf * (typename Matrix::ElementType)pow(v, 1.0_mf); };

    auto expectDy = [](typename Matrix::ElementType v) { return 2.0_mf; };

    ASSERT_FLOAT_EQ((dx[0, 0]), expectDx(1.0_mf));
    ASSERT_FLOAT_EQ((dx[0, 1]), expectDx(1.1_mf));
    ASSERT_FLOAT_EQ((dx[0, 2]), expectDx(1.2_mf));
    ASSERT_FLOAT_EQ((dx[1, 0]), expectDx(2.0_mf));
    ASSERT_FLOAT_EQ((dx[1, 1]), expectDx(2.2_mf));
    ASSERT_FLOAT_EQ((dx[1, 2]), expectDx(2.3_mf));

    ASSERT_FLOAT_EQ((dy[0, 0]), expectDy(1.0_mf));
    ASSERT_FLOAT_EQ((dy[0, 1]), expectDy(1.1_mf));
    ASSERT_FLOAT_EQ((dy[0, 2]), expectDy(1.2_mf));
    ASSERT_FLOAT_EQ((dy[1, 0]), expectDy(2.0_mf));
    ASSERT_FLOAT_EQ((dy[1, 1]), expectDy(2.2_mf));
    ASSERT_FLOAT_EQ((dy[1, 2]), expectDy(2.3_mf));
}