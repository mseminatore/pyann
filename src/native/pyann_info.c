/* Build-time info exported for pyann's backend detection. */

#ifndef PYANN_BLAS_NAME
#define PYANN_BLAS_NAME NULL
#endif

#ifdef _WIN32
#define PYANN_EXPORT __declspec(dllexport)
#else
#define PYANN_EXPORT __attribute__((visibility("default")))
#endif

#include <stddef.h>

/* Name of the BLAS libann was built against ("cblas", "openblas", "accelerate"), or NULL */
PYANN_EXPORT const char *pyann_blas_backend(void)
{
    return PYANN_BLAS_NAME;
}

/* GPU backend compiled into libann ("metal", "cuda", "vulkan"), or NULL; with
   several, the first ann_gpu_init() tries. libann exports only its ANN_API
   functions, so its backend symbols can't be probed. */
PYANN_EXPORT const char *pyann_gpu_backend(void)
{
#if defined(USE_METAL)
    return "metal";
#elif defined(USE_CUDA)
    return "cuda";
#elif defined(USE_VULKAN)
    return "vulkan";
#else
    return NULL;
#endif
}
