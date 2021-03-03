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
#include <stdlib.h>
#include <string>
#include <vector>

#include "scan_stl.hpp"
#include "scan_v1.hpp"
#include "scan_v2.hpp"
#include "scan_v3.hpp"
#include "test.hpp"

template<class T>
void exclusiveScanSerial(const T* in, T* out, std::size_t num_elements)
{
    stl::exclusive_scan(in, in+num_elements, out, size_t(0));
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

template<class T>
void exclusiveScanParallelInplace(const T* in, T* out, std::size_t num_elements)
{
    v1::exclusiveScan(out, num_elements);
}

int main(int argc, char** argv)
{
    std::size_t numElements = 10000000;
    if (argc > 1)
        numElements = std::stoi(argv[1]);

    std::cout << "scanning " << numElements << " elements\n";

    std::vector<unsigned> reference(numElements);
    std::iota(begin(reference), end(reference), 0);

    unsigned* input  = (unsigned*)aligned_alloc(4096, numElements * sizeof(unsigned));
    unsigned* output = (unsigned*)aligned_alloc(4096, numElements * sizeof(unsigned));

    int numThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        numThreads = omp_get_num_threads();
    }

    #pragma omp parallel for schedule(static, numElements/numThreads) // favors v3
    //#pragma omp parallel for schedule(static, 6144)                 // favors v1
    for (size_t i = 0; i < numElements; ++i)
    {
        input[i]  = 1;
        output[i] = 1;
    }

    test_scan("serial", input, output, numElements, reference, exclusiveScanSerial<unsigned>);
    std::copy(input, input+numElements, output);

    test_scan("serial inplace", input, output, numElements, reference, exclusiveScanSerialInplace<unsigned>);
    std::copy(input, input+numElements, output);

    test_scan("parallel v1", input, output, numElements, reference, v1::exclusiveScan<unsigned>);
    std::copy(input, input+numElements, output);

    test_scan("parallel v1 inplace", input, output, numElements, reference, exclusiveScanParallelInplace<unsigned>);
    std::copy(input, input+numElements, output);

    test_scan("parallel v2", input, output, numElements, reference, v2::exclusiveScan<unsigned>);
    std::copy(input, input+numElements, output);

    test_scan("parallel v3", input, output, numElements, reference, v3::exclusiveScan<unsigned>);

    benchmark_scan("serial", input, output, numElements, reference, exclusiveScanSerial<unsigned>);
    benchmark_scan("serial inplace", input, output, numElements, reference, exclusiveScanSerialInplace<unsigned>);
    benchmark_scan("parallel v1", input, output, numElements, reference, v1::exclusiveScan<unsigned>);
    benchmark_scan("parallel v1 inplace", input, output, numElements, reference, exclusiveScanParallelInplace<unsigned>);
    benchmark_scan("parallel v2", input, output, numElements, reference, v2::exclusiveScan<unsigned>);
    benchmark_scan("parallel v3", input, output, numElements, reference, v3::exclusiveScan<unsigned>);

    free(input);
    free(output);
}
