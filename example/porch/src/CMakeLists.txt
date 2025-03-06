add_library(porch)
target_sources(porch PUBLIC FILE_SET CXX_MODULES FILES
    backend.cpp
    module.cpp
    tensor.cpp
    size.cpp
)
target_link_libraries(porch PUBLIC
    cpp_matrix
)