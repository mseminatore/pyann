"""C declarations for ann.h"""

ANN_CDEF = """
// Forward declarations
typedef struct Network Network;
typedef struct Network *PNetwork;
typedef struct Layer Layer;
typedef struct Layer *PLayer;

// Layer types
typedef enum {
    LAYER_INPUT,
    LAYER_HIDDEN,
    LAYER_OUTPUT
} Layer_type;

// Activation types
typedef enum {
    ACTIVATION_NULL,
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU,
    ACTIVATION_LEAKY_RELU,
    ACTIVATION_TANH,
    ACTIVATION_SOFTSIGN,
    ACTIVATION_SOFTMAX
} Activation_type;

// Loss function types
typedef enum {
    LOSS_MSE,
    LOSS_CATEGORICAL_CROSS_ENTROPY
} Loss_type;

// Optimizer types
typedef enum {
    OPT_SGD,
    OPT_MOMENTUM,
    OPT_RMSPROP,
    OPT_ADAGRAD,
    OPT_ADAM
} Optimizer_type;

// Weight initialization types
typedef enum {
    WEIGHT_INIT_UNIFORM,
    WEIGHT_INIT_XAVIER,
    WEIGHT_INIT_HE,
    WEIGHT_INIT_AUTO
} Weight_init_type;

// Error codes
#define ERR_FAIL    -1
#define ERR_OK      0
#define ERR_NULL_PTR -2
#define ERR_ALLOC   -3
#define ERR_INVALID -4
#define ERR_IO      -5

// CSV flags
#define CSV_HAS_HEADER 1
#define CSV_NO_HEADER  0

// LR Scheduler callback type
typedef real(*LRSchedulerFunc)(unsigned epoch, real base_lr, void *user_data);

// Error callback type
typedef void (*ErrorLogCallback)(int error_code, const char *error_message, const char *function_name);

// Network creation and destruction
PNetwork ann_make_network(Optimizer_type opt, Loss_type loss_type);
int ann_add_layer(PNetwork pnet, int node_count, Layer_type layer_type, Activation_type activation_type);
void ann_free_network(PNetwork pnet);

// Data loading
int ann_load_csv(const char *filename, int has_header, real **data, int *rows, int *stride);

// Network save/load (text format)
int ann_save_network(const PNetwork pnet, const char *filename);
PNetwork ann_load_network(const char *filename);

// Network save/load (binary format)
int ann_save_network_binary(const PNetwork pnet, const char *filename);
PNetwork ann_load_network_binary(const char *filename);

// ONNX export/import
int ann_export_onnx(const PNetwork pnet, const char *filename);
PNetwork ann_import_onnx(const char *filename);

// Training and inference
int ann_train_network(PNetwork pnet, PTensor x_train, PTensor y_train, int rows);
int ann_train_begin(PNetwork pnet);
real ann_train_step(PNetwork pnet, const real *inputs, const real *targets, int batch_size);
void ann_train_end(PNetwork pnet);
int ann_predict(PNetwork pnet, const real *inputs, real *outputs);
real ann_evaluate_accuracy(PNetwork pnet, PTensor x_test, PTensor y_test);

// Network configuration
void ann_set_learning_rate(PNetwork pnet, real lr);
void ann_set_convergence(PNetwork pnet, real epsilon);
void ann_set_loss_function(PNetwork pnet, Loss_type loss);
void ann_set_batch_size(PNetwork pnet, unsigned batch_size);
void ann_set_epoch_limit(PNetwork pnet, unsigned epochs);
void ann_set_gradient_clip(PNetwork pnet, real max_gradient);
void ann_set_dropout(PNetwork pnet, real rate);
void ann_set_layer_dropout(PNetwork pnet, int layer_index, real rate);
void ann_set_training_mode(PNetwork pnet, int is_training);
void ann_set_weight_init(PNetwork pnet, Weight_init_type init_type);

// Regularization
void ann_set_weight_decay(PNetwork pnet, real lambda);
void ann_set_l1_regularization(PNetwork pnet, real lambda);

// Learning rate schedulers
void ann_set_lr_scheduler(PNetwork pnet, LRSchedulerFunc scheduler, void *user_data);

// Built-in scheduler parameter structs
typedef struct { int step_size; real gamma; } LRStepParams;
typedef struct { real gamma; } LRExponentialParams;
typedef struct { int T_max; real min_lr; } LRCosineParams;

// Built-in scheduler functions
real ann_lr_scheduler_step(unsigned epoch, real base_lr, void *data);
real ann_lr_scheduler_exponential(unsigned epoch, real base_lr, void *data);
real ann_lr_scheduler_cosine(unsigned epoch, real base_lr, void *data);

// Network info
int ann_get_layer_count(const PNetwork pnet);
int ann_get_layer_nodes(const PNetwork pnet, int layer_index);
Activation_type ann_get_layer_activation(const PNetwork pnet, int layer_index);

// Confusion matrix
real ann_confusion_matrix(PNetwork pnet, PTensor inputs, PTensor outputs, int *tp, int *fp, int *tn, int *fn);
void ann_print_confusion_matrix(PNetwork pnet, PTensor inputs, PTensor outputs);
int ann_class_prediction(PNetwork pnet, const real *outputs);

// Visualization and export
int ann_export_pikchr(const PNetwork pnet, const char *filename);
int ann_export_learning_curve(const PNetwork pnet, const char *filename);
void ann_clear_history(PNetwork pnet);

// Debug and info
void ann_print_props(const PNetwork pnet);
void ann_print_outputs(const PNetwork pnet);

// Error handling
const char* ann_strerror(int error_code);
void ann_set_error_log_callback(ErrorLogCallback callback);
ErrorLogCallback ann_get_error_log_callback(void);
void ann_clear_error_log_callback(void);
"""
