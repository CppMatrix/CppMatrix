#ifndef MATRIX_TEST_NAME
#error MATRIX_TEST_NAME must be defined.
#endif

#include <gtest/gtest.h>

#define XSTR(x) STR(x)
#define STR(x) #x
#define CONN2(X, Y) X##Y
#define CONN(X, Y) CONN2(X, Y)
#define LINEAR_TEST_NAME CONN(MATRIX_TEST_NAME, _NeuralNetwork_Modules_LinearTest)

class LINEAR_TEST_NAME : public testing::Test {
public:
    void SetUp() override
    {
        if (!Matrix::IsAvaliable()) {
            GTEST_SKIP() << XSTR(MATRIX_TEST_NAME) << " is not avaliable, skip.";
        }
    }
};

#define LINEAR_TEST(X) TEST_F(LINEAR_TEST_NAME, X)

import cpp_matrix.neural_network;

using namespace cpp_matrix;
using Linear = cpp_matrix::neural_network::Linear<Matrix>;

LINEAR_TEST(LinearSize)
{
    auto m = Linear(20, 30);
    auto input = Matrix::Random(128, 20);
    auto output = m(input);
    ASSERT_EQ(output.Row(), 128);
    ASSERT_EQ(output.Column(), 30);
}