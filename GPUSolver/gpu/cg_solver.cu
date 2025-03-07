#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#define DEFAULT_FPX double
#if (__cplusplus >= 201703L)  ///< if c++17 or above
#define TEMPLATE_WITH_TYPE_T template<typename T = DEFAULT_FPX>
#else
#define TEMPLATE_WITH_TYPE_T template<typename T>
#endif


/* ================================================================================================
 *  ERROR CHECKING
 * ================================================================================================ */
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

    cusparseHandle_t &cuSolverHandle() { return m_sparseHandle; }
};


/* ================================================================================================
 *  DSparseCSRMatrix (CSR SPARSE MATRIX)
 * ================================================================================================ */

TEMPLATE_WITH_TYPE_T
class DSparseCSRMatrix {
private:
    /* Metadata */
    size_t m_numRows = 0;  ///< Number of rows
    size_t m_numCols = 0;  ///< Number of columns
    size_t m_numNonZeros = 0;  ///< Number of nonzero elements

    /* Data */
    T *m_d_data = nullptr;  ///< Pointer to device data
    int *m_d_csrOffsets = nullptr;
    int *m_d_csrColumns = nullptr;
    cusparseSpMatDescr_t m_csrMat;

public:
    DSparseCSRMatrix(const std::vector<T> &data,
                     const std::vector<int> &csrOffsets,
                     const std::vector<int> &csrColumns,
                     size_t nRows,
                     size_t nCols,
                     size_t nNonzero) :
            m_numCols(nCols), m_numRows(nRows), m_numNonZeros(nNonzero) {
        /* allocate memory */
        cudaMalloc((void **) &m_d_csrOffsets, (m_numRows + 1) * sizeof(int));
        cudaMalloc((void **) &m_d_csrColumns, m_numNonZeros * sizeof(int));
        cudaMalloc((void **) &m_d_data, m_numNonZeros * sizeof(T));
        /* copy data to device */
        gpuErrChk(cudaMemcpy(m_d_data, data.data(), m_numNonZeros * sizeof(T), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(m_d_csrColumns, csrColumns.data(), m_numNonZeros * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(m_d_csrOffsets, csrOffsets.data(), (m_numRows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        /* create CSR */
        cusparseCreateCsr(&m_csrMat, m_numRows, m_numCols, m_numNonZeros,
                          m_d_csrOffsets, m_d_csrColumns, m_d_data,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
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
    }

};


/* ================================================================================================
 *  MAIN function (for testing only)
 * ================================================================================================ */

int main(void) {
    /* SESSION */

    /* SPARSE MATRIX A */
    const int num_rows = 4;
    const int num_cols = 4;
    const int nnz = 9;
    std::vector<int> h_csrOffsets{0, 3, 4, 7, 9};
    std::vector<int> h_columns{0, 2, 3, 1, 0, 2, 3, 1, 3};
    std::vector<float> h_values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                6.0f, 7.0f, 8.0f, 9.0f};

    DSparseCSRMatrix<float>(h_values, h_csrOffsets, h_columns, num_rows, num_cols, nnz);
    std::cout << "hello\n";
}