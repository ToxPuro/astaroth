#include "type_conversion.h"

#include "errchk.h"

int
test_type_conversion(void)
{
    int retval = 0;

    retval |= WARNCHK(can_convert<int32_t>(std::numeric_limits<int32_t>::max()) == true);
    retval |= WARNCHK(can_convert<int32_t>(std::numeric_limits<int32_t>::min()) == true);
    retval |= WARNCHK(can_convert<int64_t>(std::numeric_limits<int32_t>::max()) == true);
    retval |= WARNCHK(can_convert<int64_t>(std::numeric_limits<int32_t>::min()) == true);
    retval |= WARNCHK(can_convert<int32_t>(std::numeric_limits<int64_t>::max()) == false);
    retval |= WARNCHK(can_convert<int32_t>(std::numeric_limits<int64_t>::min()) == false);

    retval |= WARNCHK(can_convert<uint32_t>(std::numeric_limits<uint32_t>::max()) == true);
    retval |= WARNCHK(can_convert<uint32_t>(std::numeric_limits<uint32_t>::min()) == true);
    retval |= WARNCHK(can_convert<uint64_t>(std::numeric_limits<uint32_t>::max()) == true);
    retval |= WARNCHK(can_convert<uint64_t>(std::numeric_limits<uint32_t>::min()) == true);
    retval |= WARNCHK(can_convert<uint32_t>(std::numeric_limits<uint64_t>::max()) == false);
    retval |= WARNCHK(can_convert<uint32_t>(std::numeric_limits<uint64_t>::min()) == true);

    retval |= WARNCHK(can_convert<int32_t>(std::numeric_limits<uint32_t>::max()) == false);
    retval |= WARNCHK(can_convert<int32_t>(std::numeric_limits<uint32_t>::min()) == true);
    retval |= WARNCHK(can_convert<int64_t>(std::numeric_limits<uint32_t>::max()) == true);
    retval |= WARNCHK(can_convert<int64_t>(std::numeric_limits<uint32_t>::min()) == true);
    retval |= WARNCHK(can_convert<int32_t>(std::numeric_limits<uint64_t>::max()) == false);
    retval |= WARNCHK(can_convert<int32_t>(std::numeric_limits<uint64_t>::min()) == true);

    return retval;
}
