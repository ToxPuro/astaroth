#include "ntuple.h"
#include "type_conversion.h"

#include "print.h"

#include <array>

template <typename T, size_t N>
std::array<T, N>
operator+(const std::array<T, N>& a, const std::array<T, N>& b)
{
    std::array<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, std::array<T, N>>::type
operator+(const T& scal, const std::array<T, N>& arr)
{
    std::array<T, N> result;
    std::transform(arr.begin(), arr.end(), result.begin(), [scal](T elem) { return scal + elem; });
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, std::array<T, N>>::type
operator+(const std::array<T, N>& arr, const T& scal)
{
    return scal + arr;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, std::array<T, N>>::type
operator-(const std::array<T, N>& a, const std::array<T, N>& b)
{
    std::array<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<T>());
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, std::array<T, N>>::type
operator-(const std::array<T, N>& arr, const T& scal)
{
    std::array<T, N> result;
    std::transform(arr.begin(), arr.end(), result.begin(), [scal](T elem) { return elem - scal; });
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, std::array<T, N>>::type
operator-(const T& scal, const std::array<T, N>& arr)
{
    std::array<T, N> result;
    std::transform(arr.begin(), arr.end(), result.begin(), [scal](T elem) { return scal - elem; });
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, std::array<T, N>>::type
operator*(const std::array<T, N>& a, const std::array<T, N>& b)
{
    std::array<T, N> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::multiplies<T>());
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, std::array<T, N>>::type
operator*(const T& scal, const std::array<T, N>& arr)
{
    std::array<T, N> result;
    std::transform(arr.begin(), arr.end(), result.begin(), [scal](T elem) { return scal * elem; });
    return result;
}

template <typename T, size_t N>
typename std::enable_if<std::is_arithmetic<T>::value, std::array<T, N>>::type
operator*(const std::array<T, N>& arr, const T scal)
{
    return scal * arr;
}

int
main(void)
{
    test_type_conversion();

    // std::array<size_t, ndims> arr = {1, 2, 3};
    // PRINTD(arr.size());

    // const auto arr2 = arr;
    // arr[0]++;

    // for (const auto& e : arr2)
    //     std::cout << e << " ";
    // std::cout << std::endl;

    const size_t ndims = 3;

    std::array<size_t, ndims> a = {1, 2, 3};
    std::array<size_t, ndims> b = {4, 5, 6};

    return EXIT_SUCCESS;
}
