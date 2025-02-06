//
// Created by Chelsea De Marseilla on 01/02/2025.
//

#include <iostream>
#include <optional>
#include <string>
#include <cmath>
#include <cassert>
#include <chrono>
#include <tuple>

// __global__ void compute_temperature_at_tau_v2(
//                                       const double* current_row,
//                                       double* next_row,
//                                       size_t tau,
//                                       size_t nodes_x)
// {
// }

// __global__ void compute_temperature_at_tau(double* d_data,
//                                       size_t tau,
//                                       size_t nodes_x)
// {
//     double* current_row = d_data + tau * nodes_x;
//     double* next_row = current_row + tau * nodes_x;
//     int tid = blocktid.x * blockDim.x + threadtid.x;
//     if (tid <= nodes_x - 2)
//     {
//         // UNROLLING
//         int tid = 2*tid + 1;
//         double x1 = current_row[tid+1];
//         double x2 = current_row[tid];
//         next_row[tid] = x2 + x1 - 3*current_row[tid-1];
//         next_row[tid+1] = x1 + current_row[tid+2] - 3*x2;
//     }
// }

// __global__ void solve_explicit_parallel(double* dev_grid, double sigma, double rate, int s_nodes, double dS, double dt)
// {
//
//     int tid = blocktid.x * blockDim.x + threadtid.x;
//     if(tid <= s_nodes - 2){
//         int tid = tid + 1; // sets tid to start from 1
//
//         double current_s = tid * dS;
//
//         // calculating greeks
//         double delta = (current_col[tid+1] - current_col[tid-1]) /  (2 * dS);
//         double gamma = (current_col[tid+1] - 2 * current_col[tid] + current_col[tid-1]) / pow(dS, 2);
//         double theta = -0.5 * pow(sigma, 2) * pow(current_s, 2) * gamma - rate * current_s * delta + rate * current_col[tid];
//         next_col[tid] = current_col[tid] - (theta * dt);
//     }
// }

__global__ void solve_explicit_parallel(const double* current_col, double* next_col, double sigma, double rate, int s_nodes, double dS, double dt)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=1 && tid < s_nodes){

	    double current_s = tid * dS;

	    // calculating greeks
	    double delta = (current_col[tid+1] - current_col[tid-1]) /  (2 * dS);
	    double gamma = (current_col[tid+1] - 2 * current_col[tid] + current_col[tid-1]) / pow(dS, 2);
	    double theta = -0.5 * pow(sigma, 2) * pow(current_s, 2) * gamma - rate * current_s * delta + rate * current_col[tid];
	    next_col[tid] = current_col[tid] - (theta * dt);
	}

}

std::tuple<double, double> set_boundary_conditions(std::string option_type, double s_max, double strike_price, double rate, double expiry, int tau, double dt) {
    double lower_boundary = 0;
    double upper_boundary = 0;

    double current_time_step = tau * dt;

    if (option_type == "call") {
        upper_boundary = s_max - strike_price * std::exp(-rate * (expiry - current_time_step));
    }else if (option_type == "put") {
        lower_boundary = strike_price * std::exp(-rate * (expiry - current_time_step));
    }

    return  std::make_tuple(lower_boundary, upper_boundary);
}

static void set_terminal_condition(double* grid, std::string option_type, int s_nodes, int t_nodes, double dS,
                                   double strike_price)
{
    if (option_type == "call")
    {
        for (int i = 0; i <= s_nodes; i++)
        {
            double current_s = i * dS;
            grid[i * t_nodes + (t_nodes-1)] = fmax(current_s - strike_price, 0);
        }
    }
    else if (option_type == "put")
    {
        for (int i = 0; i <= s_nodes; i++)
        {
            double current_s = i * dS;
            grid[i * t_nodes + (t_nodes-1)] = fmax(strike_price - current_s, 0);
        }
    }
}

int main()
{
    std::string option_type = "call";
    double s_max = 300.0;
    int expiry = 1;
    double sigma = 0.2;
    double rate = 0.05;
    double strike_price = 100;
    int s_nodes = 100;
    int t_nodes = 100000;

    double dt_max = 1 / (pow(s_nodes, 2) * pow(sigma, 2));
    double dS = s_max / s_nodes;
    double dt;

    // calculates appropriate dt value (is it okay to put this here?)
    dt = static_cast<double>(expiry) / t_nodes;
    assert(dt < dt_max);

    size_t grid_size = (s_nodes + 1) * (t_nodes + 1);

    // host memory allocation
    double* host_grid = (double*)malloc(grid_size * sizeof(double));
    double* host_current = (double*)malloc((s_nodes + 1) * sizeof(double));

    // gpu memory allocation
    double* d_current, *d_next;
    cudaMalloc(&d_current, (s_nodes + 1) * sizeof(double));
    cudaMalloc(&d_next, (s_nodes + 1) * sizeof(double));

    int threads_per_block = 256;
    int num_blocks = (s_nodes + threads_per_block - 1) / threads_per_block;

    auto start = std::chrono::high_resolution_clock::now();

    // initialising host_grid values
        set_terminal_condition(host_grid, option_type, s_nodes, t_nodes, dS, strike_price);

    for (int i=0; i<=s_nodes; i++)
    {
        host_current[i] = host_grid[i * t_nodes + (t_nodes - 1)];
    }

    // copy host memory to gpu memory ( last column only)
    cudaMemcpy(d_current, host_current, (s_nodes + 1) * sizeof(double), cudaMemcpyHostToDevice);

    for (int i=t_nodes-1;i>=0;i--)
    {
        // kernel invocation - this solves and computes for the next col
        solve_explicit_parallel<<<num_blocks, threads_per_block>>>(d_current, d_next, sigma, rate, s_nodes, dS, dt);
        cudaDeviceSynchronize();
        cudaMemcpy(host_current, d_next, (s_nodes + 1) * sizeof(double), cudaMemcpyDeviceToHost);

        // setting boundary conditions
        int lower, upper;
        std::tie(lower, upper) = set_boundary_conditions(option_type, s_max, strike_price, rate, expiry, i, dt);

        host_current[0] = lower;
        host_current[s_nodes] = upper;

        //stores it in the host grid (i feel like this is a bit inefficient)
        for (int j=0; j<=s_nodes; j++)
        {
            host_grid[j * t_nodes + i] = host_current[j];
        }

        d_current = d_next;
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Duration of GPU Performance: " << duration.count() << " microseconds" << std::endl;

    // for (int i = 0; i < s_nodes + 1; i++) {
    //     for (int j = 0; j < t_nodes + 1; j++) {
    //         std::cout << host_grid[i * t_nodes + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    cudaFree(d_current);
    cudaFree(d_next);
    free(host_grid);
    free(host_current);

    return 0;

}

