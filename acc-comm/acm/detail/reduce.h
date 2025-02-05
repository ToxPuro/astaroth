#pragma once

enum class ReduceType { sum, max, min };

template <typename T> class ReduceTask {
  private:
    // Local memory

  public:
    ReduceTask(const size_t count) {}
};

void test_reduce();
