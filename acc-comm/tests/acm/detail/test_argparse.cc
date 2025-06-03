#include <cstdlib>
#include <stdexcept>

#include "acm/detail/argparse.h"
#include "acm/detail/ntuple.h"
#include "acm/detail/print_debug.h"

struct CustomArguments {
    ac::shape global_nn;
    size_t    some_counter{0};

    CustomArguments(const std::vector<std::string>& args)
    {
        auto pairs{ac::parse_key_value_pairs(args)};
        for (const auto& pair : pairs) {
            if (pair.first == "global_nn")
                global_nn = ac::parse_shape(pair.second);
            else if (pair.first == "some_counter")
                some_counter = std::stoul(pair.second);
            else
                throw std::runtime_error("Invalid argument pair \"" + pair.first + ": " +
                                         pair.second + "\"");
        }
    }
};

int
main()
{

    {
        const char* argv[] = {
            "program_name",
            "--global_nn",
            "1,2,3,4",
            "--some_counter",
            "5",
        };
        const int argc = sizeof(argv) / sizeof(argv[0]);

        std::vector<std::string> args(argv + 1, argv + argc);
        CustomArguments          arg_s{args};

        ERRCHK((arg_s.global_nn == ac::shape{1, 2, 3, 4}));
        ERRCHK((arg_s.some_counter == 5));
    }

    {
        const char* argv[] = {"program_name"};
        const int   argc   = sizeof(argv) / sizeof(argv[0]);

        std::vector<std::string> args(argv + 1, argv + argc);
        CustomArguments          arg_s{args};

        ERRCHK((arg_s.some_counter == 0));
        ERRCHK((arg_s.global_nn.size() == 0));
    }

    return EXIT_SUCCESS;
}
