/// @file
module;

export module cpp_matrix.neural_network:loss_functions;
import :backend;

namespace cpp_matrix::neural_network {

/// @brief Measure the element-wise mean squared error.
/// @param input Predicted values.
/// @param target Ground truth values.
/// @return Mean Squared Error loss, should be a 1x1 matrix.
export template <Backend Matrix>
Matrix MeanSquaredErrorLoss(const Matrix& input, const Matrix& target)
{
    return (input - target).Pow(2).Sum() / (typename Matrix::ElementType)(input.Row() * input.Column());
}

}