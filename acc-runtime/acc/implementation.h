#pragma once

// IMPLEMENTATION        : defined as acc-runtime compile option
// MAX_THREADS_PER_BLOCK : defined as acc-runtime compile option

// Implementations
#define IMPLICIT_CACHING (1)
#define EXPLICIT_CACHING (2)
#define EXPLICIT_CACHING_3D_BLOCKING (3)
#define EXPLICIT_CACHING_4D_BLOCKING (4)
#define EXPLICIT_PINGPONG_txw (5)
#define EXPLICIT_PINGPONG_txy (6)
#define EXPLICIT_ROLLING_PINGPONG (7)

#define EXPLICIT_ROLLING_PINGPONG_BLOCKSIZE                                    \
  (NUM_FIELDS <= 4 ? NUM_FIELDS : 4) // Must be less than NUM_FIELDS

#define EXPLICIT_PINGPONG_txyblocked (8)
#define EXPLICIT_PINGPONG_txyz (9)
