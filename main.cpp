/*
 * MIT License
 *
 * Copyright (c) 2021 Sebastian Keller
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! \file
 * \brief Parallel prefix sum (scan) test harness
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <chrono>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "scan_v1.hpp"
#include "scan_v2.hpp"
#include "test.hpp"

template<class T>
void exclusiveScanSerial(const T* in, T* out, std::size_t num_elements)
{
    T sum = 0;
    for (size_t i = 0; i < num_elements; ++i)
    {
        out[i] = sum;
        sum += in[i];
    }
}

template<class T>
void exclusiveScanSerialInplace(const T* in, T* out, std::size_t num_elements)
{
    T a = 0;
    T b = 0;
    for (size_t i = 0; i < num_elements; ++i)
    {
        a += out[i];
        out[i] = b;
        b = a;
    }
}

int main(int argc, char** argv)
{
    std::size_t numElements = 10000000;
    if (argc > 1)
        numElements = std::stoi(argv[1]);

    std::cout << "scanning " << numElements << " elements\n";
    std::vector<unsigned> input(numElements, 1);
    std::vector<unsigned> output(numElements);

    std::vector<unsigned> reference(numElements);
    std::iota(begin(reference), end(reference), 0);

    test_scan("serial", input, output, reference, exclusiveScanSerial<unsigned>);

    output = input;
    test_scan("serial inplace", input, output, reference, exclusiveScanSerialInplace<unsigned>);

    output = input;
    test_scan("parallel v1", input, output, reference, v1::exclusiveScan<unsigned>);

    output = input;
    test_scan("parallel v2", input, output, reference, v2::exclusiveScan<unsigned>);

    benchmark_scan("serial", input, output, reference, exclusiveScanSerial<unsigned>);
    benchmark_scan("serial inplace", input, output, reference, exclusiveScanSerialInplace<unsigned>);
    benchmark_scan("parallel v1", input, output, reference, v1::exclusiveScan<unsigned>);
    benchmark_scan("parallel v2", input, output, reference, v2::exclusiveScan<unsigned>);
}