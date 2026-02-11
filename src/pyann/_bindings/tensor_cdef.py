"""C declarations for tensor.h"""

TENSOR_CDEF = """
// Real type (float by default in libann)
typedef float real;

// Tensor structure
typedef struct {
    int rows, cols;
    int stride;
    real *values;
    int rank;
} Tensor, *PTensor;

// Transpose flags
typedef enum {
    Tensor_NoTranspose,
    Tensor_Transpose
} TENSOR_TRANSPOSE;

// Tensor creation and destruction
PTensor tensor_create(int rows, int cols);
PTensor tensor_create_from_array(int rows, int cols, const real *vals);
void tensor_set_from_array(PTensor t, int rows, int cols, const real *array);
void tensor_free(PTensor t);
PTensor tensor_ones(int rows, int cols);
PTensor tensor_zeros(int rows, int cols);
PTensor tensor_create_random_uniform(int rows, int cols, real min, real max);
PTensor tensor_onehot(const PTensor t, int classes);
PTensor tensor_copy(const PTensor t);

// Element-wise and scalar operations
PTensor tensor_add_scalar(PTensor t, real val);
PTensor tensor_add(const PTensor a, const PTensor b);
PTensor tensor_sub(const PTensor a, const PTensor b);
PTensor tensor_mul_scalar(PTensor t, real val);
PTensor tensor_mul(const PTensor a, const PTensor b);
PTensor tensor_div(const PTensor a, const PTensor b);

// Matrix operations
PTensor tensor_matvec(TENSOR_TRANSPOSE trans, real alpha, const PTensor mtx, real beta, const PTensor v, PTensor dest);
PTensor tensor_square(PTensor t);
PTensor tensor_exp(PTensor t);
PTensor tensor_argmax(const PTensor t);
PTensor tensor_max(const PTensor t);
real tensor_sum(const PTensor t);
PTensor tensor_axpy(real alpha, const PTensor x, PTensor y);
PTensor tensor_gemm(real alpha, const PTensor A, const PTensor B, real beta, PTensor C);
PTensor tensor_gemm_transB(real alpha, const PTensor A, const PTensor B, real beta, PTensor C);
PTensor tensor_gemm_transA(real alpha, const PTensor A, const PTensor B, real beta, PTensor C);
PTensor tensor_axpby(real alpha, const PTensor x, real beta, PTensor y);
PTensor tensor_outer(real alpha, const PTensor a, const PTensor b, PTensor dest);
PTensor tensor_heaviside(const PTensor a);

// Element access
real tensor_get_element(const PTensor t, int row, int col);
void tensor_set_element(PTensor t, int row, int col, real val);
PTensor tensor_slice_rows(const PTensor t, int rows);
PTensor tensor_slice_cols(const PTensor t, int cols);
void tensor_fill(PTensor t, real val);
void tensor_random_uniform(PTensor t, real min, real max);
void tensor_random_normal(PTensor t, real mean, real std);
void tensor_clip(PTensor t, real min_val, real max_val);

// I/O
void tensor_print(const PTensor t);
int tensor_save_to_file(const PTensor t, const char *filename);
"""
