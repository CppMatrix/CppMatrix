module;

#include <stdfloat>
#include <type_traits>

import cpp_matrix;

export module cpp_matrix.neural_network:backend;

namespace cpp_matrix::neural_network {

export template <typename T>
concept Backend = std::is_same_v<T, cpp_matrix::CpuMatrix<std::float16_t>>
    || std::is_same_v<T, cpp_matrix::CpuMatrix<std::float32_t>>
    || std::is_same_v<T, cpp_matrix::CudaMatrix<std::float16_t>>
    || std::is_same_v<T, cpp_matrix::CudaMatrix<std::float32_t>>;

}