/// @file
module;

#include <memory>

export module cpp_matrix.nn:modules_linear;
import :modules_module;

namespace cpp_matrix::nn {

export template <Backend Matrix>
class Linear : public Module<Matrix> {
    struct Impl : public IModule<Matrix> {
        Impl(size_t inputFeatures, size_t outputFeatures)
            : weight { Matrix::Random(inputFeatures, outputFeatures) }
            , bias { Matrix::Random(1, outputFeatures) }
        {
        }

        Matrix Forward(const Matrix& input) override
        {
            return (input * weight).AddToRow(bias);
        }

        Matrix weight {};
        Matrix bias {};
    };

public:
    Linear(auto&&... args)
        : Module<Matrix> { std::make_shared<Impl>(std::forward<decltype(args)>(args)...) }
    {
    }

    Matrix& weight()
    {
        return impl().weight;
    }

    Matrix& bias()
    {
        return impl().bias;
    }

private:
    Impl& impl()
    {
        return *(Impl*)this->m_pModule.get();
    }
};

}
