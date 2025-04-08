#pragma once

#include <vector>

#include "segment.h"

/**
 * Partition the domain mm into segments that surround the halo-less domain nn offset by nn_offset
 */
[[nodiscard]] std::vector<ac::segment> partition(const ac::shape& mm, const ac::shape& nn,
                                                 const ac::index& nn_offset);

/**
 * Prune segments inside or outside the bounding box of size nn at offset rr.
 * If the `invert_selection` flag is false (default), prunes segments that fall inside the bounding
 * box.
 * If the `invert_selection` flag is true, prunes segments that fall outside the bounding box.
 *
 * Raises an exception if there is a segment that expands beyond the boundaries.
 *
 * Visualization: x marks the area of pruned segments. Segments marked with o are returned.
 *
 * `invert_selection` flag false (default)
 *     n
 *   |----|
 * |o|xxxx|o|
 *   ^
 *   r
 *
 * `invert_selection` flag true
 *     n
 *   |----|
 * |x|oooo|x|
 *   ^
 *   r
 *
 */
[[nodiscard]] std::vector<ac::segment> prune(const std::vector<ac::segment>& segments,
                                             const ac::shape& nn, const ac::index& rr,
                                             const bool invert_selection = false);
