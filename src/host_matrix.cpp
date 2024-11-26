module;

#include <stdexcept>
#include <vector>

export module cpp_matrix:host_matrix;

namespace cpp_matrix {

export class HostMatrix {
public:
    HostMatrix() = default;

    HostMatrix(size_t row, size_t column)
        : m_row { row }
        , m_column { column }
        , m_data(row * column)
    {
    }

    size_t Row() const
    {
        return m_row;
    }

    size_t Column() const
    {
        return m_column;
    }

    HostMatrix& operator=(std::vector<float> data)
    {
        m_row = 1;
        m_column = data.size();
        m_data = std::move(data);
        return *this;
    }

    void Write(std::vector<float> data)
    {
        if (m_row * m_column != data.size()) {
            throw std::runtime_error { "Elements size is not the same." };
        }

        m_data = std::move(data);
    }

    std::vector<float> Read() const
    {
        return m_data;
    }

    HostMatrix operator*(const HostMatrix& other) const
    {
        // TODO
        return {};
    }

    float operator[](size_t row, size_t column) const
    {
        if (row >= m_row || column >= m_column) {
            throw std::runtime_error { "Out of range" };
        }

        return m_data[row * m_column + column];
    }

private:
    size_t m_row {};
    size_t m_column {};
    std::vector<float> m_data;
};

}