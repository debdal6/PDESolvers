//
// Created by Chelsea De Marseilla on 05/03/2025.
//

#ifndef BSE_SOLVERS_PARALLEL_CUH
#define BSE_SOLVERS_PARALLEL_CUH

#include <iostream>
#include <chrono>
#include <iomanip>
#include <cusparse.h>

/**
 * Define defaults
 */
#define DEFAULT_FPX double
#define THREADS_PER_BLOCK 512

/**
 * Determines the number of blocks needed for a given number of tasks, n,
 * and number of threads per block
 *
 * @param n problem size
 * @param threads_per_block threads per block (defaults to THREADS_PER_BLOCK)
 * @return number of blocks
 */
constexpr size_t numBlocks(size_t n, size_t threads_per_block = THREADS_PER_BLOCK) {
    return (n / threads_per_block + (n % threads_per_block != 0));
}

/**
 * Check for errors when calling GPU functions
 */
#define gpuErrChk(status) { gpuAssert((status), __FILE__, __LINE__); } while(false)

template<typename T = DEFAULT_FPX>
inline void gpuAssert(T code, const char *file, int line, bool abort = true) {
    if constexpr (std::is_same_v<T, cudaError_t>) {
        if (code != cudaSuccess) {
            std::cerr << "cuda error. String: " << cudaGetErrorString(code)
                      << ", file: " << file << ", line: " << line << "\n";
            if (abort) exit(code);
        }
    } else {
        std::cerr << "Error: library status parser not implemented" << "\n";
    }
}

/**
 * Option type (call or put)
 */
enum class OptionType {
    Call,
    Put
};

/**
 * Sets the terminal condition of the Black-Scholes equation, based on option type
 *
 * @tparam type option type
 * @tparam T data type (default: double)
 * @param dev_grid device memory grid storing option values
 * @param s_nodes number of spatial nodes (asset prices)
 * @param t_nodes number of time nodes
 * @param dS grid spacing between asset prices
 * @param strike_price strike price of the option
 **/
template<OptionType type, typename T = DEFAULT_FPX>
__global__ void set_terminal_condition(T *dev_grid,
                                       int s_nodes,
                                       int t_nodes,
                                       T dS,
                                       T strike_price) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int index = (t_nodes) * (s_nodes + 1) + idx;
    if (idx < s_nodes + 1) {
        T current_s = idx * dS;
        if constexpr (type == OptionType::Call) {
            dev_grid[index] = max(current_s - strike_price, 0.0);
        } else if constexpr (type == OptionType::Put) {
            dev_grid[index] = max(strike_price - current_s, 0.0);
        }
    }
}

/**
 * Sets the boundary conditions of the Black-Scholes equation
 *
 * @tparam type option type
 * @tparam T data type (default: double)
 * @param dev_grid device memory grid storing option values
 * @param t_nodes number of time nodes
 * @param s_nodes number of spatial nodes (asset prices)
 * @param s_max the maximum price of the underlying asset
 * @param strike_price strike price of the option
 * @param rate the risk-free interest rate
 * @param expiry the time to expiry
 * @param dt grid spacing between time nodes
 *
**/
template<OptionType type, typename T = DEFAULT_FPX>
__global__ void set_boundary_conditions(T *dev_grid,
                                        int t_nodes,
                                        int s_nodes,
                                        T s_max,
                                        T strike_price,
                                        T rate,
                                        T expiry,
                                        T dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx <= t_nodes) {
        T current_time_step = idx * dt;
        if constexpr (type == OptionType::Call) {
            dev_grid[idx * (s_nodes + 1)] = 0.0;
            dev_grid[idx * (s_nodes + 1) + s_nodes] = s_max - strike_price * std::exp(
                    -rate * (expiry - current_time_step));
        } else if constexpr (type == OptionType::Put) {
            dev_grid[idx * (s_nodes + 1)] = strike_price * std::exp(-rate * (expiry - current_time_step));
            dev_grid[idx * (s_nodes + 1) + s_nodes] = 0.0;
        }
    }
}

template <typename T = DEFAULT_FPX>
__global__ static void compute_coefficients(T* alpha,
                                            T* beta,
                                            T* gamma,
                                            T sigma,
                                            T rate,
                                            int s_nodes,
                                            T dS,
                                            T dt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 0 && tid < s_nodes)
    {
        T current_s = tid * dS;
        T sigma_sq = pow(sigma, 2);
        T dS_sq_inv = 1 / pow(dS, 2);
        alpha[tid] = 0.25 * dt * (sigma_sq * pow(current_s, 2) * dS_sq_inv - rate * current_s / dS);
        beta[tid] = -dt * 0.5 * (sigma_sq * pow(current_s, 2) * dS_sq_inv + rate);
        gamma[tid] = 0.25 * dt * (sigma_sq * pow(current_s, 2) * dS_sq_inv + rate * current_s / dS);
    }
}

template <typename T = DEFAULT_FPX>
__global__ static void generate_row_offset(int* row_ptr, size_t size, size_t nnz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) row_ptr[0] = 0;
    if (tid >= 1 && tid < size - 1)
    {
        row_ptr[tid] = 3 * tid - 1;
    }
    if (tid == size - 1) row_ptr[tid] = nnz;
}

template <typename T = DEFAULT_FPX>
__global__ static void generate_col_idx(int* d_col_idx, size_t vector_size, size_t nnz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        d_col_idx[0] = 0;
        d_col_idx[1] = 1;
    }

    if (tid >= 1 && tid < vector_size - 1)
    {
        d_col_idx[3 * tid - 1] = tid - 1;
        d_col_idx[3 * tid] = tid;
        d_col_idx[3 * tid + 1] = tid + 1;
    }

    if (tid == vector_size - 1)
    {
        d_col_idx[nnz - 2] = vector_size - 2;
        d_col_idx[nnz - 1] = vector_size - 1;
    }
}

template <typename T = DEFAULT_FPX>
__global__ static void generate_values(T* d_val, T* d_alpha, T* d_beta, T* d_gamma, size_t vector_size, size_t nnz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        d_val[0] = 1 - d_beta[1];
        d_val[1] = -d_gamma[1];
    }

    if (tid >= 1 && tid < vector_size - 1)
    {
        int idx = tid + 1;
        d_val[3 * tid - 1] = -d_alpha[idx];
        d_val[3 * tid] = 1 - d_beta[idx];
        d_val[3 * tid + 1] = -d_gamma[idx];
    }

    if (tid == vector_size - 1)
    {
        d_val[nnz - 2] = -d_alpha[vector_size];
        d_val[nnz - 1] = 1 - d_beta[vector_size];
    }
}

template <typename T = DEFAULT_FPX>
__global__ static void construct_rhs(T* rhs_vector,
                                     const T* current,
                                     T* prev,
                                     T* d_alpha,
                                     T* d_beta,
                                     T* d_gamma,
                                     int vector_size,
                                     int s_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        rhs_vector[0] = d_alpha[1] * (current[0] + prev[0]) + (1 + d_beta[1]) * current[1] + d_gamma[1] * current[2];
    }

    if (tid >= 1 && tid < vector_size - 1)
    {
        int idx = tid + 1;
        rhs_vector[tid] = d_alpha[idx] * current[idx - 1] + (1 + d_beta[idx]) * current[idx] + d_gamma[idx] * current[
            idx + 1];
    }

    if (tid == vector_size - 1)
    {
        rhs_vector[vector_size - 1] = d_alpha[vector_size] * current[s_nodes - 2] + (1 + d_beta[vector_size]) *
            current[s_nodes - 1] + d_gamma[vector_size] * (current[s_nodes] + prev[s_nodes]);
    }
}

/**
 * Solves the explicit method in parallel
 *
 * @tparam T data type (default: double)
 * @param current points to the current time step
 * @param prev points to the previous time step (to be computed)
 * @param sigma the volatility of the underlying asset
 * @param rate the risk-free interest rate
 * @param s_nodes the number of spatial nodes (asset prices)
 * @param dS grid spacing between spatial nodes
 * @param dt grid spacing between time nodes
**/
template<typename T = DEFAULT_FPX>
__global__ void solve_explicit_parallel(const T *current,
                                        T *prev,
                                        T sigma,
                                        T rate,
                                        int s_nodes,
                                        T dS,
                                        T dt) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1 && tid < s_nodes) {
        T current_s = tid * dS;
        // calculating greeks
        T delta = (current[tid + 1] - current[tid - 1]) / (2 * dS);
        T gamma = (current[tid + 1] - 2 * current[tid] + current[tid - 1]) / pow(dS, 2);
        T theta = -0.5 * pow(sigma, 2) * pow(current_s, 2) * gamma
                  - rate * current_s * delta + rate * current[tid];
        prev[tid] = current[tid] - (theta * dt);
    }
}

/**
 * Solution u(x, t)
 *
 * @tparam T data type (default: double)
 */
template<typename T = DEFAULT_FPX>
class Solution {
public:
    T *m_d_data = nullptr;
    size_t m_s_nodes;
    size_t m_t_nodes;
    double m_duration = 0;

    Solution(T *data, size_t s_nodes, size_t t_nodes) {
        m_d_data = data;
        m_s_nodes = s_nodes;
        m_t_nodes = t_nodes;
    };

    ~Solution() {
        if (m_d_data) {
            gpuErrChk(cudaFree(m_d_data));
            m_d_data = 0;
        }
    }

    size_t grid_size() const {
        return (m_t_nodes + 1) * (m_s_nodes + 1);
    }

    void download(T *host_data) const {
        if (m_d_data) {
            gpuErrChk(cudaMemcpy(host_data, m_d_data, grid_size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }

    // Here we can include more methods (e.g., for printing, finding average value, etc)
};

/**
 *
 * @tparam type data type (default: double)
 * @param s_max the maximum price of the underlying asset
 * @param expiry option expiry
 * @param sigma the volatility of the underlying asset
 * @param rate the risk-free interest rate
 * @param strike_price strike price
 * @param s_nodes number of nodes along S-axis
 * @param t_nodes number of nodes along T-axis
 * @return solution as instance of Solution
 */
template<OptionType type, typename T = DEFAULT_FPX>
Solution<T> solve_bse_explicit(T s_max,
                               T expiry,
                               T sigma,
                               T rate,
                               T strike_price,
                               int s_nodes,
                               int t_nodes) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t grid_size = (t_nodes + 1) * (s_nodes + 1);
    T dt_max = 1 / (pow(s_nodes, 2) * pow(sigma, 2));
    T dS = s_max / s_nodes;
    // calculates appropriate dt value (is it okay to put this here?)
    T dt = static_cast<T>(expiry) / t_nodes;
    if (dt > dt_max) throw std::runtime_error("t_nodes too small");

    // gpu memory allocation
    T *dev_grid;
    gpuErrChk(cudaMalloc(&dev_grid, grid_size * sizeof(T)));

    size_t nBlocks_s = numBlocks(s_nodes + 1);
    size_t nBlocks_t = numBlocks(t_nodes + 1);

    set_terminal_condition<type><<<nBlocks_s, THREADS_PER_BLOCK>>>(dev_grid, s_nodes, t_nodes, dS, strike_price);
    set_boundary_conditions<type><<<nBlocks_t, THREADS_PER_BLOCK>>>(dev_grid, t_nodes, s_nodes, s_max, strike_price, rate,
                                                                  expiry, dt);
    // solving equation in parallel per time step
    for (int tau = t_nodes; tau > 0; tau--) {
        T *current_time_step = dev_grid + tau * (s_nodes + 1);
        T *prev_time_step = current_time_step - (s_nodes + 1);
        solve_explicit_parallel<<<nBlocks_s, THREADS_PER_BLOCK>>>(current_time_step, prev_time_step, sigma, rate, s_nodes,
                                                                dS, dt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    Solution<T> sol(dev_grid, s_nodes, t_nodes);
    sol.m_duration = (double) duration.count() / 1e6;
    return sol;
}

template<OptionType type, typename T = DEFAULT_FPX>
Solution<T> solve_bse_cn(T s_max,
                               T expiry,
                               T sigma,
                               T rate,
                               T strike_price,
                               int s_nodes,
                               int t_nodes)
{
    auto start = std::chrono::high_resolution_clock::now();

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    size_t nBlocks_s = numBlocks(s_nodes + 1);
    size_t nBlocks_t = numBlocks(t_nodes + 1);

    size_t grid_size = (t_nodes + 1) * (s_nodes + 1);
    size_t vector_size = s_nodes - 1;
    T dt = expiry / t_nodes;
    T dS = s_max / s_nodes;

    T* dev_grid;
    gpuErrChk(cudaMalloc(&dev_grid, grid_size * sizeof(T)));

    T *d_alpha, *d_beta, *d_gamma;
    gpuErrChk(cudaMalloc(&d_alpha, (s_nodes + 1) * sizeof(T)));
    gpuErrChk(cudaMalloc(&d_beta, (s_nodes + 1) * sizeof(T)));
    gpuErrChk(cudaMalloc(&d_gamma, (s_nodes + 1) * sizeof(T)));

    compute_coefficients<<<nBlocks_s, THREADS_PER_BLOCK>>>(d_alpha, d_beta, d_gamma, sigma, rate, (s_nodes + 1), dS, dt);

    size_t nnz = 3 * vector_size - 2;

    int *d_row_ptr, *d_col_idx;
    T* d_val;
    gpuErrChk(cudaMalloc(&d_row_ptr, (vector_size + 1) * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    gpuErrChk(cudaMalloc(&d_val, nnz * sizeof(T)));

    // initialise row pointers
    generate_row_offset<<<nBlocks_s, THREADS_PER_BLOCK>>>(d_row_ptr, vector_size + 1, nnz);
    cudaDeviceSynchronize();

    // initialise column indices
    generate_col_idx<<<nBlocks_s, THREADS_PER_BLOCK>>>(d_col_idx, vector_size, nnz);
    cudaDeviceSynchronize();

    // initialise values
    generate_values<<<numBlocks(nnz), THREADS_PER_BLOCK>>>(d_val, d_alpha, d_beta, d_gamma, vector_size, nnz);
    cudaDeviceSynchronize();

    int* h_row_ptr = (int*)malloc((vector_size + 1) * sizeof(int));
    int* h_col_idx = (int*)malloc(nnz * sizeof(int));
    T* h_val = (T*)malloc(nnz * sizeof(T));

    cudaMemcpy(h_row_ptr, d_row_ptr, (vector_size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col_idx, d_col_idx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val, d_val, nnz * sizeof(double), cudaMemcpyDeviceToHost);

    // create sparse matrix in csr format
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseSpSVDescr_t spsvDescr;
    void* pBuffer = nullptr;
    size_t bufferSize = 0;
    double scalar = 1;

    cusparseCreateCsr(&matA, vector_size, vector_size, nnz,
                  d_row_ptr, d_col_idx, d_val,
                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    set_terminal_condition<type><<<nBlocks_s, THREADS_PER_BLOCK>>>(dev_grid, s_nodes, t_nodes, dS, strike_price);
    cudaDeviceSynchronize();

    set_boundary_conditions<type><<<nBlocks_t, THREADS_PER_BLOCK>>>(dev_grid, t_nodes, s_nodes, s_max, strike_price, rate,
                                                                  expiry, dt);
    cudaDeviceSynchronize();
    
    for (int tau = t_nodes; tau > 0; tau--)
    {
        // points to the current time step
        T* current = dev_grid + tau * (s_nodes + 1);
        T* prev = current - (s_nodes + 1);

        T* d_rhs_vector;
        gpuErrChk(cudaMalloc(&d_rhs_vector, vector_size * sizeof(T)));

        // constructing RHS vector
        construct_rhs<<<numBlocks(vector_size), THREADS_PER_BLOCK>>>(d_rhs_vector, current, prev, d_alpha, d_beta, d_gamma,
                                                      vector_size, s_nodes);

        T* h_rhs_vector = (T*)malloc((vector_size * sizeof(T)));
        cudaMemcpy(h_rhs_vector, d_rhs_vector, (vector_size * sizeof(T)), cudaMemcpyDeviceToHost);

            // for (int i=0; i<vector_size; i++)
            // {
            //     std::cout << h_rhs_vector[i] << " ";
            // }
            // std::cout << std::endl;

        cudaDeviceSynchronize();

        cusparseSpSV_createDescr(&spsvDescr);

        cusparseCreateDnVec(&vecX, vector_size, d_rhs_vector, CUDA_R_64F);
        cusparseCreateDnVec(&vecY, vector_size, prev + 1, CUDA_R_64F);

        cusparseSpSV_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &scalar, matA, vecX, vecY, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize);

        if (!pBuffer) gpuErrChk(cudaMalloc(&pBuffer, bufferSize));

        cusparseSpSV_analysis(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &scalar, matA, vecX, vecY, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, pBuffer);

        // this might be affecting the results (triangular solver - do i need to factorise the tridiagonal matrix?)
        cusparseSpSV_solve(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &scalar, matA, vecX, vecY, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr);

        double* h_prev = (double*)malloc((s_nodes + 1) * sizeof(double));
        cudaMemcpy(h_prev, prev, (s_nodes + 1) * sizeof(double), cudaMemcpyDeviceToHost);

        // std::cout << "Values of prev cn: ";
        // for (int i = 0; i <= s_nodes; ++i) {
        //     std::cout << h_prev[i] << " ";
        // }
        // std::cout << std::endl;


        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseSpSV_destroyDescr(spsvDescr);
        cudaFree(d_rhs_vector);
        // free(h_rhs_vector);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    Solution<T> sol(dev_grid, s_nodes, t_nodes);
    sol.m_duration = (double) duration.count() / 1e6;


    // free(h_row_ptr);
    // free(h_col_idx);
    // free(h_val);

    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_gamma);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_val);
    cudaFree(pBuffer);

    cusparseDestroy(handle);

    return sol;
}

#endif //BSE_SOLVERS_PARALLEL_CUH
