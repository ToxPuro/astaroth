#!/usr/bin/env python
# %%

types = ["size_t", "int64_t", "int"]

for a in types:
    print(f"#define as_{a}_array(count, a, b) _Generic((a), ", end="")
    for b in types:
        if a == b:
            continue
        print(f"{b}*: {b}_as_{a}_array, ", end="")
        print(f"const {b}*: {b}_as_{a}_array", end="")
        if b != types[-1]:
            print(", ", end="")
    print(")(count, a, b)")

print("---")
for a in types:
    for b in types:
        if a == b:
            continue
        print(f"void {b}_as_{a}_array(const size_t count, const {b}* a, {a}* b);")

print("---")

for a in types:
    for b in types:
        if a == b:
            continue
        print(
            f"void {b}_as_{a}_array(const size_t count, const {b}* a, {a}* b) {{ for (size_t i=0; i<count;++i) b[i] = as_{a}(a[i]); }}"
        )
