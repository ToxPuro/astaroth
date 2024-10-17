#pragma once

#include <array>
#include <iostream>
#include <vector>

/*
 * Templates for type-generic printing
 */
template <typename T>
std::ostream&
operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "{\n";
    for (size_t i = 0; i < vec.size(); ++i)
        os << "\t" << i << ": " << vec[i] << (i + 1 < vec.size() ? ",\n" : "\n}");
    return os;
}

template <typename T>
void
print(const std::string& label, const T& value)
{
    std::cout << label << ": " << value << std::endl;
}

template <typename T>
void
print_array(const std::string& label, const size_t count, const T* arr)
{
    std::cout << label << ": {";
    for (size_t i = 0; i < count; ++i)
        std::cout << arr[i] << (i + 1 < count ? ", " : "");
    std::cout << "}" << std::endl;
}

#define PRINTD(value) (print(#value, (value)))
#define PRINTD_ARRAY(count, arr) (print_array(#arr, (count), (arr)))
