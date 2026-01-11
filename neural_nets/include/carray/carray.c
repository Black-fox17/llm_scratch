#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "carray.h"


typedef enum{
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV
} Operation;

void carray_compute_strides(carray *a) {
    a->strides[a->ndim - 1] = 1;
    for (int i = (int)a->ndim - 2; i >= 0; i--) {
        a->strides[i] = a->strides[i + 1] * a->shape[i + 1];
    }
}

size_t carray_compute_size(size_t ndim, size_t *shape) {
    size_t s = 1;
    for (size_t i = 0; i < ndim; i++)
        s *= shape[i];
    return s;
}
static carray init_carray_with_data(void *data, size_t *shape, size_t ndim, bool owns) {
    carray a;
    a.data = data;
    a.ndim = ndim;
    a.owns = owns;
    for (size_t i = 0; i < ndim; i++) {
        a.shape[i] = shape[i];
    }
    a.size = carray_compute_size(ndim, shape);
    carray_compute_strides(&a);
    return a;
}

carray init_carray_with_scalar_value(size_t *shape, size_t ndim, float value) {
    size_t size = carray_compute_size(ndim, shape);
    float *data = malloc(sizeof(float) * size);
    for (size_t i = 0; i < size; i++)
        data[i] = value;
    return init_carray_with_data(data, shape, ndim, true);
}
carray init_carray_with_zeros(size_t *shape, size_t ndim) {
    return init_carray_with_scalar_value(shape, ndim, 0.0);
}
carray init_carray_with_ones(size_t *shape, size_t ndim) {
    return init_carray_with_scalar_value(shape, ndim, 1.0);
}

carray arange(size_t *shape, size_t ndim) {
    size_t size = carray_compute_size(ndim, shape);
    float *data = malloc(sizeof(float) * size);
    for (size_t i = 0; i < size; i++)
        data[i] = i;
    return init_carray_with_data(data, shape, ndim, true);
}

void carray_binary_op(carray *a, carray *b, carray *out, Operation op) {
    
}
void carray_print(carray *arr) {
    if (arr->ndim == 1) {
        printf("[");
        for (size_t i = 0; i < arr->shape[0]; i++) {
            printf("%f ", ((float*)arr->data)[i]);
        }
        printf("]\n");
        return;
    }
    if (arr->ndim == 2) {
        printf("[");
        for (size_t i = 0; i < arr->shape[0]; i++) {
            printf("[");
            for (size_t j = 0; j < arr->shape[1]; j++) {
                printf("%f ", ((float*)arr->data)[i * arr->shape[1] + j]);
            }
            printf("]\n");
        }
        printf("]\n");
        return;
    }
    if (arr->ndim == 3) {
        printf("[");
        for (size_t i = 0; i < arr->shape[0]; i++) {
            printf("[");
            for (size_t j = 0; j < arr->shape[1]; j++) {
                printf("[");
                for (size_t k = 0; k < arr->shape[2]; k++) {
                    printf("%f ", ((float*)arr->data)[i * arr->shape[1] * arr->shape[2] + j * arr->shape[2] + k]);
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
        return;
    }
    if (arr->ndim == 4) {
        printf("[");
        for (size_t i = 0; i < arr->shape[0]; i++) {
            printf("[");
            for (size_t j = 0; j < arr->shape[1]; j++) {
                printf("[");
                for (size_t k = 0; k < arr->shape[2]; k++) {
                    printf("[");
                    for (size_t l = 0; l < arr->shape[3]; l++) {
                        printf("%f ", ((float*)arr->data)[i * arr->shape[1] * arr->shape[2] * arr->shape[3] + j * arr->shape[2] * arr->shape[3] + k * arr->shape[3] + l]);
                    }
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
        return;
    }
    // printf("[");
    // for (size_t i = 0; i < arr->shape[0]; i++) {
    //   size_t cur_offset = offset + (arr->strides[cur_depth] * i);
    //   _printers[arr->dtype](arr->data, cur_offset);
    // }
    // printf("]");
    // return;
}