#pragma once

#include <iostream>
#include <vector>

template <typename T>
std::ostream&
operator<<(std::ostream& os, const std::vector<T>& vec)
{
    if (vec.size() < 8) {
        os << "{";
        for (size_t i = 0; i < vec.size(); ++i)
            os << vec[i] << (i + 1 < vec.size() ? ", " : "");
        os << "}";
    }
    else {
        os << "{\n";
        for (size_t i = 0; i < vec.size(); ++i)
            os << "\t" << i << ": " << vec[i] << (i + 1 < vec.size() ? ", " : "") << std::endl;
        os << "}";
    }
    return os;
}

template <typename T>
void
print_debug(const std::string& label, const T& value)
{
    std::cout << label << ": " << value << std::endl;
}

template <typename T>
void
print_debug_array(const std::string& label, const size_t count, const T* arr)
{
    std::cout << label << ": {";
    for (size_t i = 0; i < count; ++i)
        std::cout << arr[i] << (i + 1 < count ? ", " : "");
    std::cout << "}" << std::endl;
}

#define PRINT_DEBUG(value) (print_debug(#value, (value)))
#define PRINT_DEBUG_ARRAY(count, arr) (print_debug_array(#arr, (count), (arr)))
