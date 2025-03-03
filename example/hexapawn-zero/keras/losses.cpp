module;

#include <format>
#include <functional>
#include <stdexcept>
#include <stdfloat>
#include <string_view>

import cpp_matrix;

export module keras:losses;
import :layer;

namespace keras::losses {

export template <typename Matrix>
using Loss = std::function<std::float32_t(Layer<Matrix> y_true, Layer<Matrix> y_pred)>;

export template <typename Matrix>
Loss<Matrix> CategoricalCrossentropy(bool from_logits)
{
    return [](Layer<Matrix> y_true, Layer<Matrix> y_pred) -> typename Matrix::ElementType { return {}; };
}

export template <typename Matrix>
Loss<Matrix> MeanSquaredError()
{
    return [](Layer<Matrix> y_true, Layer<Matrix> y_pred) -> typename Matrix::ElementType {
        auto data = (y_true.GetData() - y_pred.GetData()).Pow(2).Read();
        typename Matrix::ElementType sum {};
        for (auto v : data) {
            sum += v;
        }
        return sum / data.size();
    };
}

export template <typename Matrix>
Loss<Matrix> GetLossByName(std::string_view name)
{
    if (name == "mean_squared_error") {
        return MeanSquaredError<Matrix>();
    } else {
        throw std::runtime_error { std::format("Unsupported loss function: {}", name) };
    }
}

}