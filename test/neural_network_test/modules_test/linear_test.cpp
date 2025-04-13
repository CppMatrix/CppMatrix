#include "../neural_network_test_common.h"

using Linear = cpp_matrix::nn::Linear<Matrix>;

NEURAL_NETWORK_TEST(Module, LinearSize)
{
    auto m = Linear(20, 30);
    auto input = Matrix::Random(128, 20);
    auto output = m(input);
    ASSERT_EQ(output.Row(), 128);
    ASSERT_EQ(output.Column(), 30);
}