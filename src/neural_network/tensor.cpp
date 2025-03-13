/// @file
module;

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
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
        : Tensor { "", std::move(m) }
    {
    }

    /// @brief Initialize a tensor with label and matrix data.
    Tensor(std::string label, Matrix m)
        : m_label { std::move(label) }
        , m_matrix { std::move(m) }
        , m_grads { { m_label, { m_matrix.Row(), m_matrix.Column(), (ElementType)1 } } }
    {
    }

    /// @brief Initialize a tensor with init data.
    Tensor(std::initializer_list<std::initializer_list<ElementType>> initData)
        : Tensor { "", std::move(initData) }
    {
    }

    /// @brief Initialize a tensor with label and init data.
    Tensor(std::string label, std::initializer_list<std::initializer_list<ElementType>> initData)
        : m_label { std::move(label) }
        , m_matrix { std::move(initData) }
        , m_grads { { m_label, { m_matrix.Row(), m_matrix.Column(), (ElementType)1 } } }
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

    Tensor operator-() const
    {
        return *this * -1;
    }

    Tensor operator+(ElementType v) const
    {
        auto res = Tensor { m_matrix + v };
        res.m_grads = m_grads;
        return res;
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
        res.m_grads = m_grads;
        for (auto& g : res.m_grads) {
            g.second = (n * m_matrix.Pow(n - 1)).ElementProduct(g.second);
        }
        return res;
    }

    Tensor operator+(const Tensor& other) const
    {
        auto res = Tensor { m_matrix + other.m_matrix };
        for (auto& gA : m_grads) {
            auto itB = other.m_grads.find(gA.first);
            res.m_grads.emplace(gA.first, itB != other.m_grads.end() ? gA.second + itB->second : gA.second);
        }
        for (auto& gB : other.m_grads) {
            auto itA = m_grads.find(gB.first);
            if (itA == m_grads.end()) {
                res.m_grads.emplace(gB.first, gB.second);
            }
        }
        return res;
    }

    Tensor operator*(ElementType n) const
    {
        auto res = Tensor { m_matrix * n };
        res.m_grads = m_grads;
        for (auto& g : res.m_grads) {
            g.second = g.second * n;
        }
        return res;
    }

    Tensor Exp() const
    {
        auto r = m_matrix.Exp();
        auto res = Tensor { m_label, r };
        res.m_grads = m_grads;
        for (auto& g : res.m_grads) {
            g.second = r.ElementProduct(g.second);
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

    /// @brief Applies the element-wise function: v / x
    Tensor DivBy(ElementType v) const
    {
        auto res = Tensor { v / m_matrix };
        res.m_grads = m_grads;
        for (auto& g : res.m_grads) {
            g.second = (-v / m_matrix.Pow(2)).ElementProduct(g.second);
        }
        return res;
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

    /// @brief Compute gradients.
    Tensor Backward() const
    {
        if (m_grads.size() > 1) {
            throw std::runtime_error { "Label must be specified for multiple inputs." };
        } else if (m_grads.empty()) {
            return {};
        } else {
            return m_grads.begin()->second;
        }
    }

private:
    std::string m_label {};
    Matrix m_matrix {};
    std::unordered_map<std::string, Matrix> m_grads {};
    mutable std::optional<std::vector<ElementType>> m_data {};
};

}

#define OpOperators(M, op, Func)                                                                                       \
    export M operator op(typename M::ElementType v, const M& m)                                                        \
    {                                                                                                                  \
        return m.Func(v);                                                                                              \
    }

#define ReverseOpOperators(M, op)                                                                                      \
    export M operator op(typename M::ElementType v, const M& m)                                                        \
    {                                                                                                                  \
        return m.operator op(v);                                                                                       \
    }

// clang-format off
#define Operators(M)            \
    ReverseOpOperators(M, +)    \
    ReverseOpOperators(M, *)    \
    OpOperators(M, /, DivBy)
//    OpOperators(M, -)            \

// clang-format on

Operators(cpp_matrix::neural_network::Tensor<cpp_matrix::CpuMatrix<std::float16_t>>);
Operators(cpp_matrix::neural_network::Tensor<cpp_matrix::CpuMatrix<std::float32_t>>);
Operators(cpp_matrix::neural_network::Tensor<cpp_matrix::CudaMatrix<std::float16_t>>);
Operators(cpp_matrix::neural_network::Tensor<cpp_matrix::CudaMatrix<std::float32_t>>);
