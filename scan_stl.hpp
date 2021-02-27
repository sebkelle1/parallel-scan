
#pragma once

#include <type_traits>

namespace stl
{

template<class InputIterator, class OutputIterator, class T>
void exclusive_scan(InputIterator in1, InputIterator in2, OutputIterator out, T init)
{
    while (in1 != in2)
    {
        *out++ = init;
        init += *in1++;
    }
}

} // namespace stl
