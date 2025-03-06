module;

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

export module porch:size;

namespace porch {

/// @brief Size is the result type of a call to Tensor::Size(). It describes the size of all dimensions of the
///        original tensor. It supports common sequence operations like indexing and length.
export class Size final {
public:
    /// @brief Create a Size object with 0 dimension (scalar type).
    Size() = default;

    /// @brief Create a Size object with 1 dimension (vector type) which contains n elements.
    Size(size_t n)
        : m_dims { n, 0 }
    {
    }

    /// @brief Create a Size object with 2 dimensions (matrix type) which contains \c row rows and \c column columns
    /// elements.
    Size(size_t row, size_t column)
        : m_dims { row, column }
    {
    }

    /// @brief Return the dimensions of this Size.
    /// @return
    size_t dimensions() const
    {
        return (bool)m_dims.first + (bool)m_dims.second;
    }

    /// @brief Return the number of elements at dimension index \c index.
    /// @exception std::runtime_error If \c index is out of dimensions.
    size_t operator[](size_t index) const
    {
        if (index >= dimensions()) {
            throw std::runtime_error { "Dimension index out of range" };
        }

        return index == 0 ? m_dims.first : m_dims.second;
    }

    /// @brief Return the number of elements a Tensor with the given size would contain.
    size_t num_of_elements() const
    {
        return std::max(m_dims.first, (size_t)1) * std::max(m_dims.second, (size_t)1);
    }

private:
    std::pair<size_t, size_t> m_dims {};
};

}