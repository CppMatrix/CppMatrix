/// @file
module;

#include <atomic>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <stdfloat>
#include <unordered_map>
#include <utility>
#include <vector>

import cpp_matrix;

export module cpp_matrix.nn:tensor;
export import :backend;

namespace cpp_matrix::nn {

static std::atomic_uint32_t s_tensor_id_seed {};

/// @brief Tensor is a matrix supports auto grad.
export template <Backend Matrix>
class Tensor {
public:
    using ElementType = typename Matrix::ElementType;

    /// @brief Initialize a tensor with matrix data.
    Tensor(Matrix m)
        : m_matrix { std::move(m) }
        , m_grads { { m_id, { m_matrix.Row(), m_matrix.Column(), (ElementType)1 } } }
    {
    }

    /// @brief Initialize a tensor with init data.
    Tensor(std::initializer_list<std::initializer_list<ElementType>> initData)
        : m_matrix { std::move(initData) }
        , m_grads { { m_id, { m_matrix.Row(), m_matrix.Column(), (ElementType)1 } } }
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
        auto res = Tensor { m_matrix - other.m_matrix };
        res.m_grads.clear();
        for (auto& gA : m_grads) {
            auto itB = other.m_grads.find(gA.first);
            res.m_grads.emplace(gA.first, itB != other.m_grads.end() ? gA.second - itB->second : gA.second);
        }
        for (auto& gB : other.m_grads) {
            auto itA = m_grads.find(gB.first);
            if (itA == m_grads.end()) {
                res.m_grads.emplace(gB.first, -gB.second);
            }
        }
        return res;
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
        res.m_grads.clear();
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

    Tensor operator/(ElementType n) const
    {
        auto res = Tensor { m_matrix / n };
        res.m_grads = m_grads;
        for (auto& g : res.m_grads) {
            g.second = g.second / n;
        }
        return res;
    }

    Tensor Exp() const
    {
        auto r = m_matrix.Exp();
        auto res = Tensor { r };
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

    /// @brief Returns the sum of all elements in the tensor.
    /// @return Should be a \a 1x1 Tensor.
    Tensor Sum() const
    {
        auto res = Tensor { m_matrix.Sum() };
        res.m_grads = m_grads;
        return res;
    }

    /** @brief Get derivative of tensor at \c x.
    
        This function will return derivative of tensor at \c x.

        For example:
        \code{.cpp}
        auto x = Tensor {
            { 1, 2, 3 },
            { 4, 5, 6 },
        };

        auto y = Tensor {
            { 0.1, 0.2, 0.3 },
            { 0.4, 0.5, 0.6 },
        };

        auto f = 3*x + 4*y;
        auto dx = f.Derivative(x);
        auto dy = f.Derivative(y);
        \endcode

        Because:

        \f[
        \begin{aligned}
                    f &= 3 \cdot x + 4 \cdot y \\\\
        {d \over dx}f &= {d \over dx}(3 \cdot x) + {d \over dx}(4 \cdot y) \\
                      &= 3 \cdot {d \over dx}x + 4 \cdot {d \over dx}y \\
                      &= 3 \\\\
        {d \over dy}f &= {d \over dy}(3 \cdot x) + {d \over dy}(4 \cdot y) \\
                      &= 3 \cdot {d \over dy}x + 4 \cdot {d \over dy}y \\
                      &= 4
        \end{aligned}
        \f]

        So after executing, \c f, \c dx and \c dy should be:
        \f[
        \begin{aligned}
        f &= 
        \begin{bmatrix}
             3.4 &  6.8 & 10.2 \\
            13.6 & 17.0 & 20.4
        \end{bmatrix}
        \\\\
        dx &=
        \begin{bmatrix}
            3 & 3 & 3 \\
            3 & 3 & 3
        \end{bmatrix}
        \\\\
        dy &=
        \begin{bmatrix}
            4 & 4 & 4 \\
            4 & 4 & 4
        \end{bmatrix}
        \end{aligned}
        \f]
    */
    Tensor Derivative(const Tensor& x) const
    {
        auto it = m_grads.find(x.m_id);
        if (it == m_grads.end()) {
            return {};
        } else {
            return it->second;
        }
    }

    /// @brief Returns the sum of all elements in the tensor.
    /// @return Should be a \a 1x1 Tensor.
    Tensor SumX() const
    {
        auto res = Tensor { m_matrix.Sum() };
        res.m_grads = m_grads;
        return res;
    }


private:
    const std::atomic_uint32_t m_id { s_tensor_id_seed++ };
    Matrix m_matrix {};
    std::unordered_map<uint32_t, Matrix> m_grads {};
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

Operators(cpp_matrix::nn::Tensor<cpp_matrix::CpuMatrix<std::float16_t>>);
Operators(cpp_matrix::nn::Tensor<cpp_matrix::CpuMatrix<std::float32_t>>);
Operators(cpp_matrix::nn::Tensor<cpp_matrix::CudaMatrix<std::float16_t>>);
Operators(cpp_matrix::nn::Tensor<cpp_matrix::CudaMatrix<std::float32_t>>);
