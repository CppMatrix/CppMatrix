module;

export module cpp_matrix.neural_network:loss_functions;
import :backend;

namespace cpp_matrix::neural_network {

/// @brief Creates a criterion that measures the mean squared error between each element in the input matrix
///        \c x and the target matrix \c y .
export template <Backend Matrix>
Matrix MeanSquaredErrorLoss(const Matrix& input, const Matrix& target)
{
    return (input - target).Pow(2).Sum() / (typename Matrix::ElementType)(input.Row() * input.Column());
}

}