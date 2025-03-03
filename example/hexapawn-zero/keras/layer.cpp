module;

#include <functional>
#include <memory>

import cpp_matrix;

export module keras:layer;

namespace keras {

export template <typename Matrix>
class ILayer {
public:
    virtual Matrix& GetData() = 0;
};

export template <typename Matrix>
class Layer {
    using ILayer = ILayer<Matrix>;

public:
    Layer() = default;

    Layer(std::shared_ptr<ILayer> p)
        : m_p { std::move(p) }
    {
    }

    Matrix& GetData()
    {
        return m_p->GetData();
    }

protected:
    std::shared_ptr<ILayer> m_p {};
};

}