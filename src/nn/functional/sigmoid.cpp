///@ file
module;

export module cpp_matrix.nn:functional_sigmoid;
import :tensor;

namespace cpp_matrix::nn::functional {

/** @brief Applies the element-wise function \f$ Sigmoid(x) \f$

    \f$ Sigmoid(x) \f$ is defined as:
    \f[

        Sigmoid(x) = {1 \over 1 + e^{-x}}

    \f]
*/
export template <Backend Matrix>
Tensor<Matrix> sigmoid(const Tensor<Matrix>& input)
{
    return (typename Matrix::ElementType)1 / ((typename Matrix::ElementType)1 + (-input).Exp());
}

}