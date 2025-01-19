#pragma once

#include <cxxabi.h>
#include <iostream>
#include <memory>
#include <vector>

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
    for (size_t i{0}; i < count; ++i)
        std::cout << arr[i] << (i + 1 < count ? ", " : "");
    std::cout << "}" << std::endl;
}

namespace print::debug {
static inline void
indent(const size_t depth)
{
    for (size_t i = 0; i < depth; ++i)
        std::cout << "    ";
}

template <typename T>
static inline void
print(const T& elem)
{
    std::cout << elem;
}

template <typename T>
static inline void
print(const std::vector<T>& vec, const size_t depth = 0)
{
    print::debug::indent(depth);
    std::cout << "{ ";
    for (const auto& elem : vec)
        std::cout << elem << " ";
    std::cout << "}" << std::endl;
}

template <typename T>
static inline void
print(const std::vector<std::vector<T>>& vec, const size_t depth = 0)
{
    print::debug::indent(depth);
    std::cout << "{" << std::endl;
    for (const auto& elem : vec) {
        print(elem, depth + 1);
    }
    print::debug::indent(depth);
    std::cout << "}" << std::endl;
}
} // namespace print::debug

template <typename T>
static inline void
print_debug(const std::string& label, const std::vector<T>& vec)
{
    std::cout << label << ":" << std::endl;
    print::debug::print(vec);
}

#define PRINT_DEBUG(value) (print_debug(#value, (value)))
#define PRINT_DEBUG_ARRAY(count, arr) (print_debug_array(#arr, (count), (arr)))
#define PRINT_DEBUG_VECTOR(vec) (print_debug(#vec, (vec)))

template <typename T>
void
print_demangled(const T& obj)
{
    int status;
    std::unique_ptr<char, void (*)(char*)> res{abi::__cxa_demangle(typeid(obj).name(),
                                                                   nullptr,
                                                                   nullptr,
                                                                   &status),
                                               [](char* ptr) { std::free(ptr); }};
    std::cout << "Type: " << (status == 0 ? res.get() : typeid(obj).name()) << std::endl;
}
