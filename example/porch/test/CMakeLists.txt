set(SRCS
    size_test.cpp
    tensor_test.cpp
)

macro(add_porch_test backend_name backend_type)
    add_executable(porch_test_${backend_name} ${SRCS})
    target_compile_definitions(porch_test_${backend_name} PRIVATE -DBACKEND_NAME=${backend_name} -DBACKEND=${backend_type})
    target_link_libraries(porch_test_${backend_name} porch GTest::gtest_main)
    gtest_discover_tests(porch_test_${backend_name})
endmacro()

add_porch_test(CpuFloat16 cpp_matrix::CpuMatrix<std::float16_t>)
add_porch_test(CpuFloat32 cpp_matrix::CpuMatrix<std::float32_t>)
add_porch_test(CudaFloat16 cpp_matrix::CudaMatrix<std::float16_t>)
add_porch_test(CudaFloat32 cpp_matrix::CudaMatrix<std::float32_t>)