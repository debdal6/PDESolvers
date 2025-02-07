//
// Created by Chelsea De Marseilla on 07/02/2025.
//

#include <iostream>
#include <chrono>
// #include "include/tensor.cuh" // how do i link GPUtils to PDESolvers???

enum class OptionType
{
    Call,
    Put
};

__global__ void set_terminal_condition(double* dev_grid, OptionType type, int s_nodes, int t_nodes, double dS,
                                       double strike_price)
{
    using enum OptionType;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int index = (t_nodes) * (s_nodes + 1) + idx;

    if (idx < s_nodes + 1)
    {
        if (type == Call)
        {
            double current_s = idx * dS;
            dev_grid[index] = max(current_s - strike_price, 0.0);
        }
        else if (type == Put)
        {
            double current_s = idx * dS;
            dev_grid[index] = max(strike_price - current_s, 0.0);
        }
    }
}

__global__ void set_boundary_conditions(double* dev_grid, int t_nodes, int s_nodes, OptionType type, double s_max,
                                        double strike_price, double rate, double expiry, double dt)
{
    using enum OptionType;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < t_nodes)
    {
        double current_time_step = idx * dt;
        if (type == Call)
        {
            dev_grid[idx * (s_nodes + 1)] = 0.0;
            dev_grid[idx * (s_nodes + 1) + s_nodes] = s_max - strike_price * std::exp(
                -rate * (expiry - current_time_step));
        }
        else if (type == Put)
        {
            dev_grid[idx * (s_nodes + 1)] = strike_price * std::exp(-rate * (expiry - current_time_step));
            dev_grid[idx * (s_nodes + 1) + s_nodes] = 0.0;
        }
    }
}

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
    OptionType type = OptionType::Call;
    double s_max = 300.0;
    int expiry = 1;
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
    cudaMalloc(&dev_grid, grid_size * sizeof(double));

    cudaMemcpy(dev_grid, host_grid, grid_size * sizeof(double), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    set_terminal_condition<<<10, 256>>>(dev_grid, type, s_nodes, t_nodes, dS, strike_price);
    set_boundary_conditions<<<10, 256>>>(dev_grid, t_nodes, s_nodes, type, s_max, strike_price, rate, expiry, dt);

    // solving equation in parallel per time step
    for (int tau = t_nodes; tau > 0; tau--)
    {
        double* current_time_step = dev_grid + tau * (s_nodes + 1);
        double* prev_time_step = current_time_step - (s_nodes + 1);
        solve_explicit_parallel<<<10, 256>>>(current_time_step, prev_time_step, sigma, rate, s_nodes, dS, dt);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // copy back into grid
    cudaMemcpy(host_grid, dev_grid, grid_size * sizeof(double), cudaMemcpyDeviceToHost);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration of GPU Performance: " << duration.count() << " microseconds" << std::endl;

    cudaFree(dev_grid);
    free(host_grid);
}
