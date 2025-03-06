#include <gtest/gtest.h>

import porch;
using namespace porch;

#define CONN(X, Y) X##Y
#define SIZE_TEST(X, Y, Z) TEST(CONN(X, Y), Z)

SIZE_TEST(SizeTest, BACKEND_NAME, DefaultSize)
{
    Size s {};
    ASSERT_EQ(s.dimensions(), 0);
    ASSERT_THROW(s[0], std::runtime_error);
    ASSERT_THROW(s[1], std::runtime_error);
    ASSERT_THROW(s[2], std::runtime_error);
    ASSERT_EQ(s.num_of_elements(), 1);
}
