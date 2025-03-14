/// @file
module;

#include <utility>

export module cpp_matrix.neural_network:functional;
import :tensor;

namespace cpp_matrix::neural_network::functional {

/** @brief Measure the element-wise <b>M</b>ean <b>S</b>quared <b>E</b>rror.

    If \f$ input \f$ is the matrix of predicted values and \f$ target \f$ is the matrix of ground truth values,
    then:

    \f[
        MSE(input, target) = {1 \over n} \sum_{i=1}^n (input_i - target_i)^2
    \f]

    @param input Predicted values.
    @param target Ground truth values.
    @return Mean Squared Error loss, should be a \a 1x1 Tensor.

*/
export template <Backend Matrix>
Tensor<Matrix> MeanSquaredErrorLoss(const Tensor<Matrix>& input, const Tensor<Matrix>& target)
{
    return (input - target).Pow(2).Sum() / (typename Matrix::ElementType)(input.Row() * input.Column());
}

/// @brief Alias of MeanSquaredErrorLoss().
export auto MseLoss(auto&&... args)
{
    return MeanSquaredErrorLoss(std::forward<decltype(args)>(args)...);
}

/** @brief Applies the element-wise function \f$ Sigmoid(x) \f$

    \f$ Sigmoid(x) \f$ is defined as:
    \f[

        Sigmoid(x) = {1 \over 1 + e^{-x}}

    \f]
*/
export template <Backend Matrix>
Tensor<Matrix> Sigmoid(const Tensor<Matrix>& tensor)
{
    return (typename Matrix::ElementType)1 / ((typename Matrix::ElementType)1 + (-tensor).Exp());
}

}