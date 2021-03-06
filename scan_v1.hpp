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

namespace v1
{

template<class T, int NPages>
void exclusiveScan(const T* in, T* out, size_t numElements)
{
    constexpr int blockSize = (NPages * 4096) / sizeof(T);
    constexpr int clSize = 64/sizeof(T);

    constexpr int maxThreads = 256;
    T* sb_ = (T*)aligned_alloc(4096, 2 * (maxThreads+1) * clSize * sizeof(T));
    T (*superBlock)[maxThreads+1][clSize] = (T (*)[maxThreads+1][clSize]) sb_;  

    int numThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        numThreads = omp_get_num_threads();

        int tid = omp_get_thread_num();

        superBlock[0][tid][0] = 0;
        superBlock[1][tid][0] = 0;
        if (tid == numThreads - 1)
        {
            superBlock[0][numThreads][0] = 0;
            superBlock[1][numThreads][0] = 0;
        }
    }

    unsigned elementsPerStep = numThreads * blockSize;
    unsigned nSteps = numElements / elementsPerStep;

    #pragma omp parallel num_threads(numThreads)
    {
        int tid = omp_get_thread_num();
        for (size_t step = 0; step < nSteps; ++step)
        {
            size_t stepOffset = step * elementsPerStep + tid * blockSize;

            stl::exclusive_scan(in + stepOffset, in + stepOffset + blockSize, out + stepOffset, 0);

            superBlock[step%2][tid][0] = out[stepOffset + blockSize - 1] + in[stepOffset + blockSize -1];

            #pragma omp barrier

            T tSum = superBlock[(step+1)%2][numThreads][0];
            for (size_t t = 0; t < tid; ++t)
                tSum += superBlock[step%2][t][0];

            if (tid == numThreads - 1)
                superBlock[step%2][numThreads][0] = tSum + superBlock[step%2][numThreads - 1][0];

            std::for_each(out + stepOffset, out + stepOffset + blockSize, [shift=tSum](T& val){ val += shift; });
        }
    }

    // remainder
    T stepSum = superBlock[(nSteps+1)%2][numThreads][0];
    stl::exclusive_scan(in + nSteps*elementsPerStep, in + numElements, out + nSteps*elementsPerStep, stepSum);

    free(sb_);
}

template<class T>
T exclusiveScanSerialInplace(T* out, size_t num_elements, T init)
{
    T a = init;
    T b = init;
    for (size_t i = 0; i < num_elements; ++i)
    {
        a += out[i];
        out[i] = b;
        b = a;
    }
    return b;
}

template<class T, int NPages>
void exclusiveScan(T* out, size_t numElements)
{
    constexpr int blockSize = (NPages * 4096) / sizeof(T);

    int numThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        numThreads = omp_get_num_threads();
    }

    T superBlock[2][numThreads+1];
    std::fill(superBlock[0], superBlock[0] + numThreads+1, 0);
    std::fill(superBlock[1], superBlock[1] + numThreads+1, 0);

    unsigned elementsPerStep = numThreads * blockSize;
    unsigned nSteps = numElements / elementsPerStep;

    #pragma omp parallel num_threads(numThreads)
    {
        int tid = omp_get_thread_num();
        for (size_t step = 0; step < nSteps; ++step)
        {
            size_t stepOffset = step * elementsPerStep + tid * blockSize;

            superBlock[step%2][tid] = exclusiveScanSerialInplace(out + stepOffset, blockSize, 0u);

            #pragma omp barrier

            T tSum = superBlock[(step+1)%2][numThreads];
            for (size_t t = 0; t < tid; ++t)
                tSum += superBlock[step%2][t];

            if (tid == numThreads - 1)
                superBlock[step%2][numThreads] = tSum + superBlock[step%2][numThreads - 1];

            std::for_each(out + stepOffset, out + stepOffset + blockSize, [shift=tSum](T& val){ val += shift; });
        }
    }

    // remainder
    T stepSum = superBlock[(nSteps+1)%2][numThreads];
    exclusiveScanSerialInplace(out + nSteps*elementsPerStep, numElements - nSteps*elementsPerStep, stepSum);
}

} // namespace v1
