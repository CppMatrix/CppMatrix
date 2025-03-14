module;

#include <cassert>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

export module cpp_matrix:cpu_matrix;
import :matrix_type;

namespace cpp_matrix::backend {

export template <MatrixElementType T>
class CpuMatrix {
    template <MatrixElementType R>
    friend CpuMatrix<R> operator-(R v, const CpuMatrix<R>& m);

    template <MatrixElementType R>
    friend CpuMatrix<R> operator/(R v, const CpuMatrix<R>& m);

public:
    using ElementType = T;

    static bool IsAvaliable()
    {
        return true;
    }

    CpuMatrix() = default;

    CpuMatrix(size_t row, size_t column)
        : m_row { row }
        , m_column { column }
        , m_data(row * column)
    {
    }

    CpuMatrix(const CpuMatrix& other)
        : m_row { other.m_row }
        , m_column { other.m_column }
        , m_data { other.m_data }
    {
    }

    CpuMatrix(CpuMatrix&& other)
        : m_row { other.m_row }
        , m_column { other.m_column }
        , m_data { std::move(other.m_data) }
    {
        other.m_row = 0;
        other.m_column = 0;
        assert(other.m_data.empty());
    }

    size_t Row() const
    {
        return m_row;
    }

    size_t Column() const
    {
        return m_column;
    }

    CpuMatrix& operator=(std::vector<T> data)
    {
        m_row = 1;
        m_column = data.size();
        m_data = std::move(data);
        return *this;
    }

    void Write(std::span<T> data)
    {
        if (m_row * m_column != data.size()) {
            throw std::runtime_error { "Elements size is not the same." };
        }

        m_data = std::vector<T> { std::begin(data), std::end(data) };
    }

    void Write(std::vector<T> data)
    {
        if (m_row * m_column != data.size()) {
            throw std::runtime_error { "Elements size is not the same." };
        }

        m_data = std::move(data);
    }

    void Write(size_t row, size_t column, const CpuMatrix& m)
    {
        assert(m_row >= row + m.m_row && m_column >= column + m.m_column);

        for (auto r = 0u; r < m.m_row; ++r) {
            memcpy(m_data.data() + (r + row) * m_column + column, m.m_data.data() + r * m.m_column,
                sizeof(ElementType) * m.m_column);
        }
    }

    std::vector<T> Read() const
    {
        return m_data;
    }

    CpuMatrix operator+(const CpuMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        CpuMatrix res { m_row, m_column };
        auto* pR = res.m_data.data();
        auto* p1 = m_data.data();
        auto* p2 = other.m_data.data();
        for (auto i = 0u; i < m_row * m_column; ++i) {
            *pR++ = *p1++ + *p2++;
        }
        return res;
    }

    CpuMatrix& operator+=(const CpuMatrix& other)
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        auto* p1 = m_data.data();
        auto* p2 = other.m_data.data();
        for (auto i = 0u; i < m_row * m_column; ++i) {
            *p1++ += *p2++;
        }
        return *this;
    }

    CpuMatrix operator+(ElementType v) const
    {
        CpuMatrix res { m_row, m_column };
        for (auto i = 0u; i < m_row * m_column; ++i) {
            res.m_data[i] = m_data[i] + v;
        }
        return res;
    }

    CpuMatrix operator-(const CpuMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        CpuMatrix res { m_row, m_column };
        auto* pR = res.m_data.data();
        auto* p1 = m_data.data();
        auto* p2 = other.m_data.data();
        for (auto i = 0u; i < m_row * m_column; ++i) {
            *pR++ = *p1++ - *p2++;
        }
        return res;
    }

    CpuMatrix operator*(ElementType v) const
    {
        CpuMatrix res { m_row, m_column };
        for (auto i = 0u; i < m_row * m_column; ++i) {
            res.m_data[i] = m_data[i] * v;
        }
        return res;
    }

    CpuMatrix operator*(const CpuMatrix& other) const
    {
        CpuMatrix res { m_row, other.m_column };
        for (auto y = 0; y < m_row; ++y) {
            for (auto x = 0; x < other.m_column; ++x) {
                T sum = {};
                for (auto i = 0; i < m_column; ++i) {
                    sum += m_data[y * m_column + i] * other.m_data[i * other.m_column + x];
                }
                res.m_data[y * other.m_column + x] = sum;
            }
        }
        return res;
    }

    CpuMatrix operator/(ElementType v) const
    {
        CpuMatrix res { m_row, m_column };
        for (auto i = 0u; i < m_row * m_column; ++i) {
            res.m_data[i] = m_data[i] / v;
        }
        return res;
    }

    CpuMatrix& operator=(CpuMatrix&& other)
    {
        m_row = other.m_row;
        m_column = other.m_column;
        m_data = std::move(other.m_data);
        other.m_row = 0;
        other.m_column = 0;
        assert(other.m_data.empty());
        return *this;
    }

    CpuMatrix Transpose() const
    {
        CpuMatrix res { m_column, m_row };
        for (int c = 0u; c < m_column; ++c) {
            for (int r = 0u; r < m_row; ++r) {
                res.m_data[c * m_row + r] = m_data[r * m_column + c];
            }
        }
        return res;
    }

    CpuMatrix ElementProduct(const CpuMatrix& other) const
    {
        if (m_row != other.m_row || m_column != other.m_column) {
            throw std::runtime_error { "Shape is not the same." };
        }

        CpuMatrix res { m_row, m_column };
        auto* pR = res.m_data.data();
        auto* p1 = m_data.data();
        auto* p2 = other.m_data.data();
        for (auto i = 0u; i < m_row * m_column; ++i) {
            *pR++ = *p1++ * *p2++;
        }
        return res;
    }

    CpuMatrix Relu() const
    {
        CpuMatrix res { m_row, m_column };
        const auto* p = m_data.data();
        auto* pR = res.m_data.data();
        for (auto i = 0; i < m_row * m_column; ++i) {
            *pR++ = std::max((T)0, *p++);
        }
        return res;
    }

    CpuMatrix Exp() const
    {
        CpuMatrix res { m_row, m_column };
        for (auto i = 0u; i < m_row * m_column; ++i) {
            res.m_data[i] = std::exp((float)m_data[i]);
        }
        return res;
    }

    CpuMatrix Pow(T e) const
    {
        CpuMatrix res { m_row, m_column };
        for (auto i = 0u; i < m_row * m_column; ++i) {
            res.m_data[i] = pow(m_data[i], e);
        }
        return res;
    }

    T operator[](size_t row, size_t column) const
    {
        if (row >= m_row || column >= m_column) {
            throw std::runtime_error { "Out of range" };
        }

        return m_data[row * m_column + column];
    }

    size_t BufferSize() const
    {
        return sizeof(T) * m_row * m_column;
    }

    CpuMatrix Sum(bool byRow, bool byColumn) const
    {
        assert(byRow || byColumn);

        if (byRow && byColumn) {
            auto sum = ElementType {};
            for (auto v : m_data) {
                sum += v;
            }
            CpuMatrix res { 1, 1 };
            res.Write({ sum });
            return res;
        }
        if (byRow) {
            std::vector<ElementType> data(m_row);
            for (auto r = 0u; r < m_row; ++r) {
                auto sum = ElementType {};
                for (auto c = 0u; c < m_column; ++c) {
                    sum += m_data[r * m_column + c];
                }
                data[r] = sum;
            }
            CpuMatrix res { m_row, 1 };
            res.Write(std::move(data));
            return res;
        } else if (byColumn) {
            std::vector<ElementType> data(m_column);
            for (auto c = 0u; c < m_column; ++c) {
                auto sum = ElementType {};
                for (auto r = 0u; r < m_row; ++r) {
                    sum += m_data[r * m_column + c];
                }
                data[c] = sum;
            }
            CpuMatrix res { 1, m_column };
            res.Write(std::move(data));
            return res;
        } else {
            throw std::runtime_error { "'byRow' and 'byColumn' can't be false at the same time" };
        }
    }

    CpuMatrix AddToRow(const CpuMatrix& m) const
    {
        assert(m_column == m.m_column && m.m_row == 1);

        CpuMatrix res { m_row, m_column };
        for (auto r = 0u; r < m_row; ++r) {
            for (auto c = 0u; c < m_column; ++c) {
                res.m_data[r * m_column + c] = m_data[r * m_column + c] + m.m_data[c];
            }
        }
        return res;
    }

private:
    size_t m_row {};
    size_t m_column {};
    std::vector<T> m_data;
};

export template <MatrixElementType T>
CpuMatrix<T> operator-(T v, const CpuMatrix<T>& m)
{
    CpuMatrix<T> res { m.m_row, m.m_column };
    auto* pR = res.m_data.data();
    auto* p1 = m.m_data.data();
    for (auto i = 0u; i < m.m_row * m.m_column; ++i) {
        *pR++ = v - *p1++;
    }
    return res;
}

export template <MatrixElementType T>
CpuMatrix<T> operator/(T v, const CpuMatrix<T>& m)
{
    CpuMatrix<T> res { m.m_row, m.m_column };
    auto* pR = res.m_data.data();
    auto* p1 = m.m_data.data();
    for (auto i = 0u; i < m.m_row * m.m_column; ++i) {
        *pR++ = v / *p1++;
    }
    return res;
}

}