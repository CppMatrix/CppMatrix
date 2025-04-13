#include "neural_network_test_common.h"

using namespace cpp_matrix::nn::functional;

NEURAL_NETWORK_TEST(Functional, MseLoss)
{
    Tensor a { {
        { 1.0_mf, 1.1_mf, 1.2_mf },
        { 2.0_mf, 2.2_mf, 2.3_mf },
    } };

    Tensor b { {
        { 1.01_mf, 1.11_mf, 1.12_mf },
        { 2.10_mf, 2.11_mf, 2.21_mf },
    } };

    auto result1 = mse_loss(a, b);
    auto result2 = mse_loss(a, b);

    auto dx1a = result1.Derivative(a);
    auto dx1b = result1.Derivative(b);
    auto dx2a = result2.Derivative(a);
    auto dx2b = result2.Derivative(b);

    auto expectedDxa = [](typename Matrix::ElementType x, typename Matrix::ElementType y) {
        return (typename Matrix::ElementType)(2 * (x - y) / 6);
    };

    auto expectedDxb = [](typename Matrix::ElementType x, typename Matrix::ElementType y) {
        return (typename Matrix::ElementType)(-2 * (x - y) / 6);
    };

    auto verify = [&](const auto& c, const auto& dxa, const auto& dxb) {
        ASSERT_EQ(c.Row(), 1);
        ASSERT_EQ(c.Column(), 1);

        Matrix::ElementType sum {};
        sum += (1.0_mf - 1.01_mf) * (1.0_mf - 1.01_mf);
        sum += (1.1_mf - 1.11_mf) * (1.1_mf - 1.11_mf);
        sum += (1.2_mf - 1.12_mf) * (1.2_mf - 1.12_mf);
        sum += (2.0_mf - 2.10_mf) * (2.0_mf - 2.10_mf);
        sum += (2.2_mf - 2.11_mf) * (2.2_mf - 2.11_mf);
        sum += (2.3_mf - 2.21_mf) * (2.3_mf - 2.21_mf);
        ASSERT_FLOAT_EQ((c[0, 0]), sum / 6);

        ASSERT_FLOAT_EQ((dxa[0, 0]), expectedDxa(1.0_mf, 1.01_mf));
        ASSERT_FLOAT_EQ((dxa[0, 1]), expectedDxa(1.1_mf, 1.11_mf));
        ASSERT_FLOAT_EQ((dxa[0, 2]), expectedDxa(1.2_mf, 1.12_mf));
        ASSERT_FLOAT_EQ((dxa[1, 0]), expectedDxa(2.0_mf, 2.10_mf));
        ASSERT_FLOAT_EQ((dxa[1, 1]), expectedDxa(2.2_mf, 2.11_mf));
        ASSERT_FLOAT_EQ((dxa[1, 2]), expectedDxa(2.3_mf, 2.21_mf));

        ASSERT_FLOAT_EQ((dxb[0, 0]), expectedDxb(1.0_mf, 1.01_mf));
        ASSERT_FLOAT_EQ((dxb[0, 1]), expectedDxb(1.1_mf, 1.11_mf));
        ASSERT_FLOAT_EQ((dxb[0, 2]), expectedDxb(1.2_mf, 1.12_mf));
        ASSERT_FLOAT_EQ((dxb[1, 0]), expectedDxb(2.0_mf, 2.10_mf));
        ASSERT_FLOAT_EQ((dxb[1, 1]), expectedDxb(2.2_mf, 2.11_mf));
        ASSERT_FLOAT_EQ((dxb[1, 2]), expectedDxb(2.3_mf, 2.21_mf));
    };

    verify(result1, dx1a, dx1b);
    verify(result2, dx2a, dx2b);
}

NEURAL_NETWORK_TEST(Functional, Sigmoid)
{
    // clang-format off
    auto x = Tensor {
        {1.16_mf},
        {0.42_mf},
        {0.62_mf}
    };
    // clang-format on

    auto z = sigmoid(x);
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

    auto dx = z.Derivative(x);
    ASSERT_EQ(dx.Row(), 3);
    ASSERT_EQ(dx.Column(), 1);
    ASSERT_FLOAT_EQ((dx[0, 0]), expectedDx(1.16_mf));
    ASSERT_FLOAT_EQ((dx[0, 1]), expectedDx(0.42_mf));
    ASSERT_FLOAT_EQ((dx[0, 2]), expectedDx(0.62_mf));
}