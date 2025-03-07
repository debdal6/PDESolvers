#include <iostream>           // Printing
#include <vector>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cublas_v2.h>

#define DEFAULT_FPX double
#if (__cplusplus >= 201703L)  ///< if c++17 or above
#define TEMPLATE_WITH_TYPE_T template<typename T = DEFAULT_FPX>
#else
#define TEMPLATE_WITH_TYPE_T template<typename T>
#endif
//
//
///* ================================================================================================
// *  ERROR CHECKING
// * ================================================================================================ */

/**
 * Check for errors when calling GPU functions
 */
#define gpuErrChk(status) { gpuAssert((status), __FILE__, __LINE__); } while(false)

TEMPLATE_WITH_TYPE_T inline void gpuAssert(T code, const char *file, int line, bool abort = true) {
    if constexpr (std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "cuda error. String: " << cudaGetErrorString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else if constexpr (std::is_same_v<T, cublasStatus_t>) {
        if (code != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublas error. Name: " << cublasGetStatusName(code)
                      << ", string: " << cublasGetStatusString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else if constexpr(std::is_same_v<T, cusparseStatus_t>) {
        if (code != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "cublas error. Code: " << cusparseGetErrorString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else {
        std::cerr << "Error: library status parser not implemented" << "\n";
    }
}



/* ================================================================================================
 *  SESSION
 * ================================================================================================ */
/**
 * Singleton for Cuda library handles
 */
class Session {
public:

    static Session &getInstance() {
        static Session instance;
        return instance;
    }

private:
    Session() {
        gpuErrChk(cublasCreate(&m_cublasHandle));
        gpuErrChk(cusparseCreate(&m_sparseHandle));
    }

    ~Session() {
        gpuErrChk(cublasDestroy(m_cublasHandle));
        gpuErrChk(cusparseDestroy(m_sparseHandle));
    }

    cublasHandle_t m_cublasHandle;
    cusparseHandle_t m_sparseHandle;


public:
    Session(Session const &) = delete;

    void operator=(Session const &) = delete;

    cublasHandle_t &cuBlasHandle() { return m_cublasHandle; }

    cusparseHandle_t &cuSpraseHandle() { return m_sparseHandle; }
};


/* ================================================================================================
 *  DSparseCSRMatrix (CSR SPARSE MATRIX)
 * ================================================================================================ */
TEMPLATE_WITH_TYPE_T
class DSparseCSRMatrix {
public:
    /* Metadata */
    size_t m_numRows = 0;  ///< Number of rows
    size_t m_numCols = 0;  ///< Number of columns
    size_t m_numNonZeros = 0;  ///< Number of nonzero elements

    /* Data */
    T *m_d_data = nullptr;  ///< Pointer to device data
    int *m_d_csrOffsets = nullptr;
    int *m_d_csrColumns = nullptr;
    cusparseSpMatDescr_t m_csrMat;

    /* Buffer */
    size_t m_bufferSize = 0;
    void *m_buffer = nullptr;

public:
    DSparseCSRMatrix(const std::vector<T> &data,
                     const std::vector<int> &csrOffsets,
                     const std::vector<int> &csrColumns,
                     size_t nRows,
                     size_t nCols,
                     size_t nNonzero) :
            m_numCols(nCols), m_numRows(nRows), m_numNonZeros(nNonzero) {
        /* allocate memory */
        cudaMalloc((void **) &m_d_data, m_numNonZeros * sizeof(T));
        cudaMalloc((void **) &m_d_csrOffsets, (m_numRows + 1) * sizeof(int));
        cudaMalloc((void **) &m_d_csrColumns, m_numNonZeros * sizeof(int));
        /* copy data to device */
        gpuErrChk(cudaMemcpy(m_d_data, data.data(), m_numNonZeros * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(m_d_csrOffsets, csrOffsets.data(), (m_numRows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(m_d_csrColumns, csrColumns.data(), m_numNonZeros * sizeof(int), cudaMemcpyHostToDevice));
        /* create CSR */
        gpuErrChk(cusparseCreateCsr(&m_csrMat, m_numRows, m_numCols, m_numNonZeros,
                                    m_d_csrOffsets, m_d_csrColumns, m_d_data,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    }

    ~DSparseCSRMatrix() {
        if (m_d_csrOffsets) {
            cudaFree(m_d_csrOffsets);
            m_d_csrOffsets = nullptr;
        }
        if (m_d_csrColumns) {
            cudaFree(m_d_csrColumns);
            m_d_csrColumns = nullptr;
        }
        if (m_d_data) {
            cudaFree(m_d_data);
            m_d_data = nullptr;
        }
        if (m_buffer) {
            cudaFree(m_buffer);
            m_buffer = nullptr;
        }
    }

    /**
     * Performs y = alpha * A * x + beta * y
     * @param y vector
     * @param x vector
     * @param alpha scalar
     * @param beta scalar
     */
    void axpby(cusparseDnVecDescr_t &y,
               cusparseDnVecDescr_t &x,
               T alpha = 1.,
               T beta = 0) {
        gpuErrChk(cusparseSpMV_bufferSize(
                Session::getInstance().cuSpraseHandle(),
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, m_csrMat, x, &beta, y, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, &m_bufferSize));
        if (!m_buffer) {
            gpuErrChk(cudaMalloc((void **) &m_buffer, m_bufferSize));
            std::cout << "m_bufferSize = " << m_bufferSize << std::endl;
        }
        float a = 1, b = 0;
        gpuErrChk(cusparseSpMV(Session::getInstance().cuSpraseHandle(),
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &a, m_csrMat, x, &b, y, CUDA_R_32F,
                               CUSPARSE_SPMV_ALG_DEFAULT, m_buffer));
    }

};


//* ================================================================================================
// *  MAIN function (for testing only)
// * ================================================================================================ */

int main(void) {
    // MATRIX A DATA (CSR)
    const int A_num_rows = 4;
    const int A_num_cols = 4;
    const int A_nnz = 9;
    std::vector<int> hA_csrOffsets{0, 3, 4, 7, 9};
    std::vector<int> hA_columns{0, 2, 3, 1, 0, 2, 3, 1, 3};
    std::vector<float>  hA_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    // VECTORS
    float hX[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hY[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float hY_result[] = {19.0f, 8.0f, 51.0f, 52.0f};

    // CSR object
    DSparseCSRMatrix<float> aCSR(hA_values,
                                 std::vector<int>{0, 3, 4, 7, 9},
                                 std::vector<int>{0, 2, 3, 1, 0, 2, 3, 1, 3},
                                 A_num_rows, A_num_cols, A_nnz);

    // VECTORS X and Y
    float  *dX, *dY;
    gpuErrChk(cudaMalloc((void **) &dX, A_num_cols * sizeof(float)));
    gpuErrChk(cudaMalloc((void **) &dY, A_num_rows * sizeof(float)));
    gpuErrChk(cudaMemcpy(dX, hX, A_num_cols * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(dY, hY, A_num_rows * sizeof(float), cudaMemcpyHostToDevice));
    cusparseDnVecDescr_t vecX, vecY;

    // Create dense vectors X and Y
    gpuErrChk(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));
    gpuErrChk(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));

    // execute SpMV
    aCSR.axpby(vecY, vecX);

    // destroy matrix/vector descriptors
    gpuErrChk(cusparseDestroyDnVec(vecX));
    gpuErrChk(cusparseDestroyDnVec(vecY));


    //--------------------------------------------------------------------------
    // device result check
    gpuErrChk(cudaMemcpy(hY, dY, A_num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        std::cout << hY[i] << std::endl;
        if (hY[i] != hY_result[i]) { // direct floating point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    if (correct)
        printf("spmv_csr_example test PASSED\n");
    else
        printf("spmv_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    gpuErrChk(cudaFree(dX));
    gpuErrChk(cudaFree(dY));

    return EXIT_SUCCESS;
}