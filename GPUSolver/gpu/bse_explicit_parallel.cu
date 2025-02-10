//
// Created by Chelsea De Marseilla on 07/02/2025.
//

#include <iostream>
#include <chrono>

/**
 * Define defaults
 */
#define DEFAULT_FPX double
#define THREADS_PER_BLOCK 512
#if (__cplusplus >= 201703L)  ///< if c++17 or above
#define TEMPLATE_WITH_TYPE_T template<typename T = DEFAULT_FPX>
#else
#define TEMPLATE_WITH_TYPE_T template<typename T>
#endif
#if (__cplusplus >= 202002L)  ///< if c++20 or above
#define TEMPLATE_CONSTRAINT_REQUIRES_FPX requires std::floating_point<T>
#else
#define TEMPLATE_CONSTRAINT_REQUIRES_FPX
#endif

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

TEMPLATE_WITH_TYPE_T inline void gpuAssert(T code, const char *file, int line, bool abort = true) {
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


enum class OptionType
{
    Call,
    Put
};

/**
 * Sets the terminal condition of the Black-Scholes equation, based on option type
 *
 * @param dev_grid device memory grid storing option values
 * @param s_nodes number of spatial nodes (asset prices)
 * @param t_nodes number of time nodes
 * @param dS grid spacing between asset prices
 * @param strike_price strike price of the option
 **/
template<OptionType type>
__global__ void set_terminal_condition(double* dev_grid, int s_nodes, int t_nodes, double dS,
                                       double strike_price)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int index = (t_nodes) * (s_nodes + 1) + idx;
    if (idx < s_nodes + 1)
    {
        double current_s = idx * dS;
        if constexpr (type == OptionType::Call)
        {
            dev_grid[index] = max(current_s - strike_price, 0.0);
        }
        else if constexpr (type == OptionType::Put)
        {
            dev_grid[index] = max(strike_price - current_s, 0.0);
        }
    }
}

/**
 * Sets the boundary conditions of the Black-Scholes equation
 *
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
template<OptionType type>
__global__ void set_boundary_conditions(double* dev_grid, int t_nodes, int s_nodes, double s_max,
                                        double strike_price, double rate, double expiry, double dt)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < t_nodes)
    {
        double current_time_step = idx * dt;
        if constexpr (type == OptionType::Call)
        {
            dev_grid[idx * (s_nodes + 1)] = 0.0;
            dev_grid[idx * (s_nodes + 1) + s_nodes] = s_max - strike_price * std::exp(
                -rate * (expiry - current_time_step));
        }
        else if constexpr (type == OptionType::Put)
        {
            dev_grid[idx * (s_nodes + 1)] = strike_price * std::exp(-rate * (expiry - current_time_step));
            dev_grid[idx * (s_nodes + 1) + s_nodes] = 0.0;
        }
    }
}

/**
 * Solves the explicit method in parallel
 *
 * @param current points to the current time step
 * @param prev points to the previous time step (to be computed)
 * @param sigma the volatility of the underlying asset
 * @param rate the risk-free interest rate
 * @param s_nodes the number of spatial nodes (asset prices)
 * @param dS grid spacing between spatial nodes
 * @param dt grid spacing between time nodes
**/
__global__ void solve_explicit_parallel(const double* current, double* prev, double sigma, double rate, int s_nodes,
                                        double dS, double dt)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1 && tid < s_nodes)
    {
        double current_s = tid * dS;

        // calculating greeks
        double delta = (current[tid + 1] - current[tid - 1]) / (2 * dS);
        double gamma = (current[tid + 1] - 2 * current[tid] + current[tid - 1]) / pow(dS, 2);
        double theta = -0.5 * pow(sigma, 2) * pow(current_s, 2) * gamma - rate * current_s * delta + rate * current[
            tid];
        prev[tid] = current[tid] - (theta * dt);
    }
}

int main()
{
    constexpr OptionType type = OptionType::Call;
    double s_max = 300.0;
    double expiry = 0.1;
    double sigma = 0.2;
    double rate = 0.05;
    double strike_price = 100;
    int s_nodes = 1000;
    int t_nodes = 100000;

    double dt_max = 1 / (pow(s_nodes, 2) * pow(sigma, 2));
    double dS = s_max / s_nodes;
    double dt;

    // calculates appropriate dt value (is it okay to put this here?)
    dt = static_cast<double>(expiry) / t_nodes;
    if (dt > dt_max) throw std::runtime_error("t_nodes too small");

    // setting up host grid
    size_t grid_size = (t_nodes + 1) * (s_nodes + 1);
    double* host_grid = (double*)malloc(grid_size * sizeof(double));

    // gpu memory allocation
    double* dev_grid;
    gpuErrChk(cudaMalloc(&dev_grid, grid_size * sizeof(double)));
    gpuErrChk(cudaMemcpy(dev_grid, host_grid, grid_size * sizeof(double), cudaMemcpyHostToDevice));

    auto start = std::chrono::high_resolution_clock::now();

    size_t num_threads = s_nodes + 1;
    set_terminal_condition<type><<<numBlocks(num_threads), THREADS_PER_BLOCK>>>(dev_grid, s_nodes, t_nodes, dS, strike_price);
    set_boundary_conditions<type><<<numBlocks(num_threads), THREADS_PER_BLOCK>>>(dev_grid, t_nodes, s_nodes, s_max, strike_price, rate, expiry, dt);

    // solving equation in parallel per time step
    for (int tau = t_nodes; tau > 0; tau--)
    {
        double* current_time_step = dev_grid + tau * (s_nodes + 1);
        double* prev_time_step = current_time_step - (s_nodes + 1);
        solve_explicit_parallel<<<numBlocks(num_threads), THREADS_PER_BLOCK>>>(current_time_step, prev_time_step, sigma, rate, s_nodes, dS, dt);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // copy back into grid
    cudaMemcpy(host_grid, dev_grid, grid_size * sizeof(double), cudaMemcpyDeviceToHost);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration of GPU Performance: " << (double) duration.count() / 1e6 << "s" << std::endl;

    cudaFree(dev_grid);
    free(host_grid);
}
