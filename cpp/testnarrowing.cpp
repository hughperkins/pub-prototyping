#include <cstdint>

struct foo {
    int32_t a;
};

void foo(int64_t val) {
    struct foo A = { val };
}
