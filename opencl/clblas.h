typedef enum clblasStatus_ {
    clblasSuccess                         = CL_SUCCESS,
    clblasInvalidValue                    = CL_INVALID_VALUE,
    clblasInvalidCommandQueue             = CL_INVALID_COMMAND_QUEUE,
    clblasInvalidContext                  = CL_INVALID_CONTEXT,
    clblasInvalidMemObject                = CL_INVALID_MEM_OBJECT,
    clblasInvalidDevice                   = CL_INVALID_DEVICE,
    clblasInvalidEventWaitList            = CL_INVALID_EVENT_WAIT_LIST,
    clblasOutOfResources                  = CL_OUT_OF_RESOURCES,
    clblasOutOfHostMemory                 = CL_OUT_OF_HOST_MEMORY,
    clblasInvalidOperation                = CL_INVALID_OPERATION,
    clblasCompilerNotAvailable            = CL_COMPILER_NOT_AVAILABLE,
    clblasBuildProgramFailure             = CL_BUILD_PROGRAM_FAILURE,
    /* Extended error codes */
    clblasNotImplemented         = -1024, /**< Functionality is not implemented */
    clblasNotInitialized,           1023      /**< clblas library is not initialized yet */
    clblasInvalidMatA,              1022 /**< Matrix A is not a valid memory object */
    clblasInvalidMatB,              1021  /**< Matrix B is not a valid memory object */
    clblasInvalidMatC,              1020  /**< Matrix C is not a valid memory object */
    clblasInvalidVecX,              1019  /**< Vector X is not a valid memory object */
    clblasInvalidVecY,              1018  /**< Vector Y is not a valid memory object */
    clblasInvalidDim,               1017  /**< An input dimension (M,N,K) is invalid */
    clblasInvalidLeadDimA,          1016  /**< Leading dimension A must not be less than the size of the first dimension */
    clblasInvalidLeadDimB,          1015  /**< Leading dimension B must not be less than the size of the second dimension */
    clblasInvalidLeadDimC,          1014  /**< Leading dimension C must not be less than the size of the third dimension */
    clblasInvalidIncX,              1013  /**< The increment for a vector X must not be 0 */
    clblasInvalidIncY,              1012  /**< The increment for a vector Y must not be 0 */
    clblasInsufficientMemMatA,      1011  /**< The memory object for Matrix A is too small */
    clblasInsufficientMemMatB,            /**< The memory object for Matrix B is too small */
    clblasInsufficientMemMatC,            /**< The memory object for Matrix C is too small */
    clblasInsufficientMemVecX,            /**< The memory object for Vector X is too small */
    clblasInsufficientMemVecY             /**< The memory object for Vector Y is too small */
} clblasStatus;
