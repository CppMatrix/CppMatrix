NEURAL_NETWORK_TEST(Tensor, AutoGrad)
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