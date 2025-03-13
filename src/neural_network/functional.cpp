/// @file
module;

export module cpp_matrix.neural_network:functional;
import :tensor;

namespace cpp_matrix::neural_network::functional {

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