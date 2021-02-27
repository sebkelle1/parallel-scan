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
 * \brief Parallel prefix sum
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdlib.h>

#include <omp.h>

#include "scan_stl.hpp"

namespace v3
{

template<class T>
void exclusiveScan(const T* in, T* out, size_t numElements)
{
    constexpr int clSize = 64/sizeof(T);

    constexpr int maxThreads = 256;
    alignas(64) T superBlock[maxThreads][clSize];

    int numThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single 
        numThreads = omp_get_num_threads();
    }

    size_t elementsPerThread = numElements / numThreads;

    #pragma omp parallel num_threads(numThreads)
    {

        int tid = omp_get_thread_num();

        size_t threadOffset = tid * elementsPerThread;
        stl::exclusive_scan(in + threadOffset, in + threadOffset + elementsPerThread, out + threadOffset, T(0));

        superBlock[tid][0] = out[threadOffset + elementsPerThread - 1] + in[threadOffset + elementsPerThread -1];

        #pragma omp barrier

        T tSum = 0;
        for (int t = 0; t < tid; ++t)
            tSum += superBlock[t][0];

        std::for_each(out + threadOffset, out + threadOffset + elementsPerThread, [shift=tSum](T& val){ val += shift; });
    }

    // remainder
    size_t nDone = numThreads * elementsPerThread;
    T stepSum = out[nDone - 1] + in[nDone - 1];
    stl::exclusive_scan(in + nDone, in + numElements, out + nDone, stepSum);
}

} // namespace v3
