NEURAL_NETWORK_TEST(Tensor, AutoGrad1)
{
    auto x = Tensor { {
        { 1.0_mf, 2.0_mf, 3.0_mf },
        { 4.0_mf, 5.0_mf, 6.0_mf },
    } };

    auto y = Tensor { {
        { 0.1_mf, 0.2_mf, 0.3_mf },
        { 0.4_mf, 0.5_mf, 0.6_mf },
    } };

    auto f = 3.0_mf * x + 4.0_mf * y;
    auto dx = f.Derivative(x);
    auto dy = f.Derivative(y);

    ASSERT_FLOAT_EQ((f[0, 0]), 3.4_mf);
    ASSERT_FLOAT_EQ((f[0, 1]), 6.8_mf);
    ASSERT_FLOAT_EQ((f[0, 2]), 10.2_mf);
    ASSERT_FLOAT_EQ((f[1, 0]), 13.6_mf);
    ASSERT_FLOAT_EQ((f[1, 1]), 17.0_mf);
    ASSERT_FLOAT_EQ((f[1, 2]), 20.4_mf);

    ASSERT_FLOAT_EQ((dx[0, 0]), 3.0_mf);
    ASSERT_FLOAT_EQ((dx[0, 1]), 3.0_mf);
    ASSERT_FLOAT_EQ((dx[0, 2]), 3.0_mf);
    ASSERT_FLOAT_EQ((dx[1, 0]), 3.0_mf);
    ASSERT_FLOAT_EQ((dx[1, 1]), 3.0_mf);
    ASSERT_FLOAT_EQ((dx[1, 2]), 3.0_mf);

    ASSERT_FLOAT_EQ((dy[0, 0]), 4.0_mf);
    ASSERT_FLOAT_EQ((dy[0, 1]), 4.0_mf);
    ASSERT_FLOAT_EQ((dy[0, 2]), 4.0_mf);
    ASSERT_FLOAT_EQ((dy[1, 0]), 4.0_mf);
    ASSERT_FLOAT_EQ((dy[1, 1]), 4.0_mf);
    ASSERT_FLOAT_EQ((dy[1, 2]), 4.0_mf);
}

NEURAL_NETWORK_TEST(Tensor, AutoGrad2)
{
    auto x = Tensor { {
        { 1.0_mf, 1.1_mf, 1.2_mf },
        { 2.0_mf, 2.2_mf, 2.3_mf },
    } };

    auto y = Tensor { {
        { 1.0_mf, 1.1_mf, 1.2_mf },
        { 2.0_mf, 2.2_mf, 2.3_mf },
    } };

    auto z = 3.0_mf * x.Pow(2) + 2.0_mf * y;
    auto dx = z.Derivative(x);
    auto dy = z.Derivative(y);

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