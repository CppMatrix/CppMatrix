add_subdirectory(webgpu)

add_library(cpp_matrix
    backend/cuda_matrix.cu
)
target_sources(cpp_matrix PUBLIC FILE_SET CXX_MODULES FILES
    backend/cpu_matrix.cpp
    backend/cuda_matrix.cpp
    backend/webgpu_matrix.cpp
    matrix_type.cpp
    matrix.cpp
    module.cpp
)
target_link_libraries(cpp_matrix PUBLIC
    webgpu
)