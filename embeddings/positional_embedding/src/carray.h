#ifndef CARRAY_H
#define CARRAY_H

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define CARRAY_MAX_DIMS 4

typedef struct {
    void  *data;                 // raw buffer
    size_t  ndim;                 // number of dimensions
    size_t  shape[CARRAY_MAX_DIMS];
    size_t  strides[CARRAY_MAX_DIMS]; // in elements, not bytes
    size_t  size;                 // total elements
    bool owns;            
} carray;


carray init_carray(void *data, size_t shape, size_t ndim, bool owns);

void carray_compute_strides(carray *a);

size_t carray_compute_size(size_t ndim, const size_t *shape);



// int carray_create(carray *a, size_t ndim, const size_t *shape) {
//     a->ndim = ndim;
//     for (size_t i = 0; i < ndim; i++)
//         a->shape[i] = shape[i];

//     a->size = carray_compute_size(ndim, shape);
//     carray_compute_strides(a);

//     a->data = malloc(a->size * sizeof(float));
//     if (!a->data) return 0;

//     a->owns_data = 1;
//     return 1;
// }

// void carray_free(carray *a) {
//     if (a->owns_data)
//         free(a->data);

//     a->data = NULL;
//     a->size = 0;
//     a->ndim = 0;
// }

// size_t carray_offset(const carray *a, const size_t *indices) {
//     size_t off = 0;
//     for (size_t i = 0; i < a->ndim; i++) {
//         off += indices[i] * a->strides[i];
//     }
//     return off;
// }

// int carray_slice0(
//     const carray *src,
//     carray *view,
//     size_t start,
//     size_t end
// ) {
//     if (start >= end || end > src->shape[0])
//         return 0;

//     *view = *src;               // shallow copy
//     view->data += start * src->strides[0];
//     view->shape[0] = end - start;
//     view->size = carray_compute_size(view->ndim, view->shape);
//     view->owns_data = 0;
//     return 1;
// }

// int carray_reshape(carray *a, size_t ndim, const size_t *shape) {
//     size_t new_size = carray_compute_size(ndim, shape);
//     if (new_size != a->size)
//         return 0;

//     a->ndim = ndim;
//     for (size_t i = 0; i < ndim; i++)
//         a->shape[i] = shape[i];

//     carray_compute_strides(a);
//     return 1;
// }

#endif //CARRAY_H