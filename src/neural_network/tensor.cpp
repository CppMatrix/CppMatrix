/// @file
module;

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdfloat>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

import cpp_matrix;

export module cpp_matrix.neural_network:tensor;
export import :backend;

namespace cpp_matrix::neural_network {

/// @brief Tensor is a matrix supports auto grad.
export template <Backend Matrix>
class Tensor {
public:
    using ElementType = typename Matrix::ElementType;

    /// @brief Initialize a tensor with matrix data.
    Tensor(Matrix m)
        : m_matrix { std::move(m) }
    {
    }

    /// @brief Initialize a tensor with label and matrix data.
    Tensor(std::string label, Matrix m)
        : m_label { std::move(label) }
        , m_matrix { std::move(m) }
    {
    }

    /// @brief Initialize a tensor with init data.
    Tensor(std::initializer_list<std::initializer_list<ElementType>> initData)
        : m_matrix { std::move(initData) }
    {
    }

    /// @brief Initialize a tensor with label and init data.
    Tensor(std::string label, std::initializer_list<std::initializer_list<ElementType>> initData)
        : m_label { std::move(label) }
        , m_matrix { std::move(initData) }
    {
    }

    Tensor(const Tensor& other)
        : m_matrix { other.m_matrix }
        , m_grads { other.m_grads }
    {
    }

    Tensor(Tensor&& other)
        : m_matrix { std::move(other.m_matrix) }
        , m_grads { std::move(other.m_grads) }
    {
    }

    size_t Row() const
    {
        return m_matrix.Row();
    }

    size_t Column() const
    {
        return m_matrix.Column();
    }

    Tensor operator-(const Tensor& other) const
    {
        return m_matrix - other.m_matrix;
    }

    Tensor operator*(const Tensor& other) const
    {
        return m_matrix * other.m_matrix;
    }

    Tensor AddToRow(const Tensor& other) const
    {
        return m_matrix.AddToRow(other.m_matrix);
    }

    Tensor Pow(ElementType n) const
    {
        auto res = Tensor { m_matrix.Pow(n) };
        res.m_grads.emplace(m_label, n * m_matrix.Pow(n - 1));
        return res;
    }

    Tensor operator+(const Tensor& other) const
    {
        auto res = Tensor { m_matrix + other.m_matrix };
        res.m_grads = m_grads;
        for (const auto& it : other.m_grads) {
            auto sit = res.m_grads.find(it.first);
            if (sit != res.m_grads.end()) {
                sit->second = sit->second + it.second;
            } else {
                res.m_grads.emplace(it.first, it.second);
            }
        }
        return res;
    }

    Tensor operator*(ElementType n) const
    {
        auto res = Tensor { m_matrix * n };
        if (m_grads.empty()) {
            std::vector<ElementType> d(m_matrix.Row() * m_matrix.Column(), n);
            res.m_grads.emplace(m_label, Matrix { m_matrix.Row(), m_matrix.Column(), { d.begin(), d.end() } });
        } else {
            res.m_grads = m_grads;
            for (auto& it : res.m_grads) {
                it.second = n * it.second;
            }
        }
        return res;
    }

    ElementType operator[](size_t row, size_t column) const
    {
        if (!m_data) {
            m_data = m_matrix.Read();
        }

        return (*m_data)[row * m_matrix.Column() + column];
    }

    /// @brief Compute gradients.
    Tensor Backward(const std::string& label) const
    {
        auto it = m_grads.find(label);
        if (it == m_grads.end()) {
            return {};
        } else {
            return it->second;
        }
    }

private:
    std::string m_label {};
    Matrix m_matrix {};
    std::unordered_map<std::string, Matrix> m_grads {};
    mutable std::optional<std::vector<ElementType>> m_data {};
};

}

export cpp_matrix::neural_network::Tensor<cpp_matrix::CpuMatrix<std::float16_t>> operator*(
    std::float16_t n, const cpp_matrix::neural_network::Tensor<cpp_matrix::CpuMatrix<std::float16_t>>& tensor)
{
    return tensor * n;
}

export cpp_matrix::neural_network::Tensor<cpp_matrix::CpuMatrix<std::float32_t>> operator*(
    std::float32_t n, const cpp_matrix::neural_network::Tensor<cpp_matrix::CpuMatrix<std::float32_t>>& tensor)
{
    return tensor * n;
}

export cpp_matrix::neural_network::Tensor<cpp_matrix::CudaMatrix<std::float16_t>> operator*(
    std::float16_t n, const cpp_matrix::neural_network::Tensor<cpp_matrix::CudaMatrix<std::float16_t>>& tensor)
{
    return tensor * n;
}

export cpp_matrix::neural_network::Tensor<cpp_matrix::CudaMatrix<std::float32_t>> operator*(
    std::float32_t n, const cpp_matrix::neural_network::Tensor<cpp_matrix::CudaMatrix<std::float32_t>>& tensor)
{
    return tensor * n;
}