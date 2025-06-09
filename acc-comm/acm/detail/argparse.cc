#include "argparse.h"

#include <sstream>

#include "errchk.h"

namespace ac {

std::map<std::string, std::string>
parse_key_value_pairs(const std::vector<std::string>& args)
{
    std::map<std::string, std::string> map;

    if (args.size() % 2 != 0)
        ERRCHK_EXPR_DESC(false, "Uneven number of input arguments.");

    for (size_t i{0}; i + 1 < args.size(); i += 2)
        map[args[i]] = args[i + 1];

    return map;
}

std::vector<std::string>
tokenize(const std::string& input, const char delimiter)
{
    std::vector<std::string> output;

    std::istringstream iss{input};
    std::string        token;
    while (std::getline(iss, token, delimiter))
        output.push_back(token);

    return output;
}

ac::shape
parse_shape(const std::string& str)
{
    auto tokens{tokenize(str, ',')};
    auto result{ac::make_shape(tokens.size(), 0)};
    for (size_t i{0}; i < tokens.size(); ++i)
        result[i] = std::stoull(tokens[i]);

    return result;
}

} // namespace ac
