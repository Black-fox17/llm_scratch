#include "carray.h"

int main() {
    carray a;
    size_t shape[2] = {3, 4};

    a = arange(shape, 2);

    carray_print(&a);
}
