/* #include <stdarg.h> */
/* #include <stdbool.h> */
/* #include <stdint.h> */
/* #include <stdlib.h> */
#include <stddef.h>


typedef struct NN NN;

typedef struct Matrix {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Matrix;

struct Matrix matrix_from_ptr(float *ptr, size_t rows, size_t cols);

void matrix_print(struct Matrix *matrix);

void nn_back_prop(struct NN *nn, const struct Matrix *ti, const struct Matrix *to);

struct NN *nn_create(size_t n_layers, const size_t *layers);

void nn_destroy(struct NN *nn);

void nn_forward(struct NN *nn);

void nn_get_output(struct NN *nn, float *output);

void nn_learn(struct NN *nn, float rate);

void nn_set_input(struct NN *nn, const float *input, size_t len);
