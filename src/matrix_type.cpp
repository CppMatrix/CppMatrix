module;

#include <cstdint>
#include <stdfloat>
#include <type_traits>

export module cpp_matrix:matrix_type;

namespace cpp_matrix {

export template <typename T>
concept MatrixElementType = std::is_same_v<T, std::float32_t> || std::is_same_v<T, std::float16_t>;

}