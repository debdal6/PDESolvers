#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// Constants
#define LENGTH_OF_ROD 10.0f
#define MAX_TIME 1.0f
#define DIFFUSIVITY 1.0f
#define NUM_POINTS_SPACE 1000
#define NUM_POINTS_TIME 2000

int main() {
    // Initialize cuSPARSE
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);

    // Calculate step sizes
    float dx = LENGTH_OF_ROD / (NUM_POINTS_SPACE - 1);
    float dt = MAX_TIME / (NUM_POINTS_TIME - 1);
    float lambda = (DIFFUSIVITY * dt) / (2.0f * dx * dx);

    // Matrix dimensions (for interior points)
    int n = NUM_POINTS_SPACE - 2;

    // Allocate host arrays
    float *h_diag = new float[n];
    float *h_upper = new float[n-1];
    float *h_lower = new float[n-1];
    float *h_temp = new float[NUM_POINTS_SPACE];
    float *h_rhs = new float[n];

    // Initialize tridiagonal matrix elements
    for(int i = 0; i < n; i++) {
        h_diag[i] = 1.0f + 2.0f * lambda;
        if(i < n-1) {
            h_upper[i] = -lambda;
            h_lower[i] = -lambda;
        }
    }

    // Initialize initial condition: u0(x) = sin(x)
    for (int i = 0; i < NUM_POINTS_SPACE; i++) {
        float x = i * dx;
        h_temp[i] = sin(x);
    }

    // Allocate device memory
    float *d_diag, *d_upper, *d_lower, *d_temp, *d_rhs;
    cudaMalloc(&d_diag, n * sizeof(float));
    cudaMalloc(&d_upper, (n-1) * sizeof(float));
    cudaMalloc(&d_lower, (n-1) * sizeof(float));
    cudaMalloc(&d_temp, NUM_POINTS_SPACE * sizeof(float));
    cudaMalloc(&d_rhs, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_diag, h_diag, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upper, h_upper, (n-1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lower, h_lower, (n-1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, h_temp, NUM_POINTS_SPACE * sizeof(float), cudaMemcpyHostToDevice);

    // Create description for tridiagonal matrix
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // Create sparse matrix in CSR format
    int nnz = 3 * n - 2; // Number of non-zero elements
    int *d_csrRowPtr, *d_csrColInd;
    float *d_csrVal;
    cudaMalloc(&d_csrRowPtr, (n + 1) * sizeof(int));
    cudaMalloc(&d_csrColInd, nnz * sizeof(int));
    cudaMalloc(&d_csrVal, nnz * sizeof(float));

    // Initialize CSR format on host
    int *h_csrRowPtr = new int[n + 1];
    int *h_csrColInd = new int[nnz];
    float *h_csrVal = new float[nnz];

    int idx = 0;
    h_csrRowPtr[0] = 0;
    for(int i = 0; i < n; i++) {
        if(i > 0) {
            h_csrColInd[idx] = i-1;
            h_csrVal[idx++] = -lambda;
        }
        h_csrColInd[idx] = i;
        h_csrVal[idx++] = 1.0f + 2.0f * lambda;
        if(i < n-1) {
            h_csrColInd[idx] = i+1;
            h_csrVal[idx++] = -lambda;
        }
        h_csrRowPtr[i+1] = idx;
    }

    // Copy CSR data to device
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Create sparse matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* d_buffer = nullptr;
    size_t bufferSize = 0;
    cusparseSpSVDescr_t spsvDescr;

    // Create sparse matrix descriptor
    cusparseCreateCsr(&matA, n, n, nnz,
                      d_csrRowPtr, d_csrColInd, d_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Time stepping loop
    for(int t = 1; t < NUM_POINTS_TIME; t++) {
        float t_val = t * dt;
        float alpha = 5.0f * t_val;  // This is the boundary condition alpha
        float beta = sin(LENGTH_OF_ROD) + 2.0f * t_val;

        cudaMemcpy(&d_temp[0], &alpha, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_temp[NUM_POINTS_SPACE-1], &beta, sizeof(float), cudaMemcpyHostToDevice);

        // Create vector descriptors
        cusparseCreateDnVec(&vecX, n, d_rhs, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, n, d_temp + 1, CUDA_R_32F);

        // Create SpSV descriptor
        cusparseSpSV_createDescr(&spsvDescr);

        // Prepare RHS
        float *d_temp_prev = d_temp + 1;
        cudaMemcpy(d_rhs, d_temp_prev, n * sizeof(float), cudaMemcpyDeviceToDevice);

        // Get buffer size and allocate
        float solve_alpha = 1.0f;  // Renamed from alpha to solve_alpha
        cusparseSpSV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &solve_alpha, matA, vecX, vecY, CUDA_R_32F,
                               CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize);
        if (d_buffer == nullptr) {
            cudaMalloc(&d_buffer, bufferSize);
        }

        // Analysis phase
        cusparseSpSV_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &solve_alpha, matA, vecX, vecY, CUDA_R_32F,
                             CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, d_buffer);

        // Solve phase
        cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &solve_alpha, matA, vecX, vecY, CUDA_R_32F,
                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr);

        // Cleanup vector descriptors
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseSpSV_destroyDescr(spsvDescr);

        if(t % 100 == 0) {
            cudaMemcpy(h_temp, d_temp, NUM_POINTS_SPACE * sizeof(float), cudaMemcpyDeviceToHost);
            printf("Time step %d completed\n", t);
        }
    }

    // Open output file
    FILE* outputFile = fopen("temperature_data.csv", "w");
    fprintf(outputFile, "x,t,temperature\n");

    // Write final state to file
    cudaMemcpy(h_temp, d_temp, NUM_POINTS_SPACE * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < NUM_POINTS_SPACE; i++) {
        float x = i * dx;
        float t = MAX_TIME;  // Final time
        fprintf(outputFile, "%f,%f,%f\n", x, t, h_temp[i]);
    }

    fclose(outputFile);
    printf("Results written to temperature_data.csv\n");

    // Clean up
    cusparseDestroy(cusparseHandle);
    cusparseDestroyMatDescr(descr);

    cudaFree(d_diag);
    cudaFree(d_upper);
    cudaFree(d_lower);
    cudaFree(d_temp);
    cudaFree(d_rhs);

    delete[] h_diag;
    delete[] h_upper;
    delete[] h_lower;
    delete[] h_temp;
    delete[] h_rhs;

    cusparseDestroySpMat(matA);
    cudaFree(d_buffer);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    delete[] h_csrRowPtr;
    delete[] h_csrColInd;
    delete[] h_csrVal;

    return 0;
}
