#include "type_conversion.h"

#include "errchk.h"

void
test_type_conversion(void)
{
    ERRCHK(can_convert<int32_t>(std::numeric_limits<int32_t>::max()) == true);
    ERRCHK(can_convert<int32_t>(std::numeric_limits<int32_t>::min()) == true);
    ERRCHK(can_convert<int64_t>(std::numeric_limits<int32_t>::max()) == true);
    ERRCHK(can_convert<int64_t>(std::numeric_limits<int32_t>::min()) == true);
    ERRCHK(can_convert<int32_t>(std::numeric_limits<int64_t>::max()) == false);
    ERRCHK(can_convert<int32_t>(std::numeric_limits<int64_t>::min()) == false);

    ERRCHK(can_convert<uint32_t>(std::numeric_limits<uint32_t>::max()) == true);
    ERRCHK(can_convert<uint32_t>(std::numeric_limits<uint32_t>::min()) == true);
    ERRCHK(can_convert<uint64_t>(std::numeric_limits<uint32_t>::max()) == true);
    ERRCHK(can_convert<uint64_t>(std::numeric_limits<uint32_t>::min()) == true);
    ERRCHK(can_convert<uint32_t>(std::numeric_limits<uint64_t>::max()) == false);
    ERRCHK(can_convert<uint32_t>(std::numeric_limits<uint64_t>::min()) == true);

    ERRCHK(can_convert<int32_t>(std::numeric_limits<uint32_t>::max()) == false);
    ERRCHK(can_convert<int32_t>(std::numeric_limits<uint32_t>::min()) == true);
    ERRCHK(can_convert<int64_t>(std::numeric_limits<uint32_t>::max()) == true);
    ERRCHK(can_convert<int64_t>(std::numeric_limits<uint32_t>::min()) == true);
    ERRCHK(can_convert<int32_t>(std::numeric_limits<uint64_t>::max()) == false);
    ERRCHK(can_convert<int32_t>(std::numeric_limits<uint64_t>::min()) == true);
}
