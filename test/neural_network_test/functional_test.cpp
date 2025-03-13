#include "neural_network_test_common.h"

using namespace cpp_matrix::neural_network::functional;

NEURAL_NETWORK_TEST(Functional, Sigmoid)
{
    // clang-format off
    auto x = Tensor {
        {1.16_mf},
        {0.42_mf},
        {0.62_mf}
    };
    // clang-format on

    auto z = Sigmoid(x);
    ASSERT_EQ(z.Row(), 3);
    ASSERT_EQ(z.Column(), 1);

    auto expected = [](typename Matrix::ElementType v) { return 1 / (1 + (Matrix::ElementType)std::exp((float)-v)); };

    ASSERT_FLOAT_EQ((z[0, 0]), expected(1.16_mf));
    ASSERT_FLOAT_EQ((z[1, 0]), expected(0.42_mf));
    ASSERT_FLOAT_EQ((z[2, 0]), expected(0.62_mf));

    auto expectedDx = [](typename Matrix::ElementType v) {
        auto f = (Matrix::ElementType)std::exp((float)-v);
        return f / (Matrix::ElementType)pow(1.0_mf + f, 2.0_mf);
    };

    auto dx = z.Backward();
    ASSERT_EQ(dx.Row(), 3);
    ASSERT_EQ(dx.Column(), 1);
    ASSERT_FLOAT_EQ((dx[0, 0]), expectedDx(1.16_mf));
    ASSERT_FLOAT_EQ((dx[0, 1]), expectedDx(0.42_mf));
    ASSERT_FLOAT_EQ((dx[0, 2]), expectedDx(0.62_mf));
}