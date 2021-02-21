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

template<class T>
void test_scan(std::string name, const std::vector<T>& input, std::vector<T>& output, const std::vector<T>& reference,
               void(*func)(const T*, T*, std::size_t))
{
    std::size_t numElements = input.size();

    auto tp0 = std::chrono::high_resolution_clock::now();
    func(input.data(), output.data(), numElements);
    auto tp1  = std::chrono::high_resolution_clock::now();

    bool pass = (output == reference);
    double t0 = std::chrono::duration<double>(tp1 - tp0).count();

    if (pass)
        //std::cout << name << " scan test: PASS, bandwidth: " << numElements * sizeof(unsigned) / (t0 * 1e6) << " MB/s\n";
        std::cout << name << " scan test: PASS\n";
    else
    {
        std::cout << name << " scan test: FAIL\n";
        if (numElements <= 100)
        {
            std::copy(begin(output), end(output), std::ostream_iterator<unsigned>(std::cout, " "));
            std::cout << std::endl;
        }
    }
}

template<class T>
void benchmark_scan(std::string name, const std::vector<T>& input, std::vector<T>& output, const std::vector<T>& reference,
                    void(*func)(const T*, T*, std::size_t))
{
    std::size_t numElements = input.size();

    int repetitions = 30;

    // warmup
    func(input.data(), output.data(), numElements);

    auto tp0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repetitions; ++i)
    {
        func(input.data(), output.data(), numElements);
    }
    auto tp1  = std::chrono::high_resolution_clock::now();

    double t0 = std::chrono::duration<double>(tp1 - tp0).count();

    std::cout << name << " benchmark bandwidth: " << numElements * sizeof(unsigned) / (t0 * 1e6) * repetitions << " MB/s\n";
}
