#include "carray.h"

int main() {
    carray a;
    size_t shape[2] = {3, 4};
    size_t ndim = 2;

    a = init_carray_with_zeros(shape, ndim);

    carray_print(&a);
}
