#pragma once

#include <iostream>
#include <numeric>
#include <string>

template <typename T>
void
print(const std::string& label, const T& value)
{
    std::cout << label << ": " << value;
}

template <typename T>
void
print(const std::string& label, const std::vector<T>& vec)
{
    std::cout << label << ": ";
    for (auto& value : vec)
        std::cout << value << " ";
    std::cout << std::endl;
}

inline size_t
prod(std::vector<size_t>& vec)
{
    size_t res = 1;
    for (auto& elem : vec)
        res *= elem;
    return res;
}

template <typename T>
void
print_recursive(const std::vector<size_t> shape, const std::vector<T>& vec)
{
    if (shape.size() == 1) {
        for (auto& value : vec)
            std::cout << value << " ";
    }
    else {
        const auto dims = shape.back();
        auto new_shape(shape);
        new_shape.pop_back();
        const auto offset = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                            std::multiplies<size_t>());
        for (size_t i = 0; i < dims; ++i) {
            print_recursive(new_shape, std::vector<T>(vec.begin() + i * offset,
                                                      vec.begin() + (i + 1) * offset));
            std::cout << std::endl;
        }
    }
}

template <typename T>
void
print(const std::string& label, const std::vector<size_t> shape, const std::vector<T>& vec)
{
    std::cout << label << ": " << std::endl;
    print_recursive(shape, vec);
    std::cout << std::endl;
}

#define PRINT_DEBUG(var) print(#var, var)
