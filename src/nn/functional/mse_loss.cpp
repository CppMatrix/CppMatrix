///@ file
module;

export module cpp_matrix.nn:functional_mse_loss;
import :tensor;

namespace cpp_matrix::nn::functional {

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
Tensor<Matrix> mse_loss(const Tensor<Matrix>& input, const Tensor<Matrix>& target)
{
    return (input - target).Pow(2).Sum() / (typename Matrix::ElementType)(input.Row() * input.Column());
}

}