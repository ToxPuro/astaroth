#pragma once
#include <map>

#include "acm/detail/ntuple.h"

namespace ac {

/**
 * Parses arguments into a map.
 * Arguments can be accessed by map.at("key").
 */
std::map<std::string, std::string> parse_key_value_pairs(const std::vector<std::string>& args);

/**
 * Splits the input string into tokens delimited by delimiter.
 */
std::vector<std::string> tokenize(const std::string& input, const char delimiter);

/**
 * Parses a string representation "a,b,c,..." into ac::shape.
 */
ac::shape parse_shape(const std::string& str);

} // namespace ac
