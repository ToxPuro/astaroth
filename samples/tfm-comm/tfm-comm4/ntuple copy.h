#pragma once
#include <cstddef>
#include <cstdint>

#include <algorithm>

#include <iostream>

constexpr size_t NTUPLE_MAX_NELEMS = 4;

struct Ntuple {
    size_t nelems;
    uint64_t elems[NTUPLE_MAX_NELEMS];

    // Constructor
    Ntuple(const size_t _nelems, const uint64_t* elems = nullptr);
    // Ntuple(const size_t _nelems, const uint64_t _fill_value = 0);
    // Ntuple(const Ntuple& ntuple);

    // Printing
    friend std::ostream& operator<<(std::ostream& os, const Ntuple& obj);
};

// Functions
Ntuple operator+(const Ntuple& a, const Ntuple& b);
Ntuple operator-(const Ntuple& a, const Ntuple& b);
Ntuple operator*(const size_t& a, const Ntuple& b);
