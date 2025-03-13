NEURAL_NETWORK_TEST(LossFunction, MeanSquaredErrorLoss)
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