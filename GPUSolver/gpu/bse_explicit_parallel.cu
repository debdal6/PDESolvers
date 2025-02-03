//
// Created by Chelsea De Marseilla on 01/02/2025.
//

#include <iostream>
#include <optional>
#include <string>
#include <cmath>
#include <assert.h>
#include <tuple>

__global__ static void solve_bse_explicit_parallel(double* grid, std::string option_type, double s_max, int expiry, double sigma, double rate, double strike_price, int s_nodes, int t_nodes, double dS, double dt)
{
    int tau = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // not entirely correct - how to reverse this the tau loop ?
    if(tau < t_nodes && i <= s_nodes && i>=1){

        // reversing time step direction
        int tau_reversed = t_nodes - tau - 1;

        double current_s = i * dS;

        // calculate greeks
        double delta = (grid[(i+1) + s_nodes * (tau_reversed+1)] - grid[(i-1) + s_nodes * (tau_reversed+1)]) /  (2 * dS);
        double gamma = (grid[(i+1) + s_nodes * (tau_reversed+1)] - 2 * grid[i + s_nodes * (tau_reversed+1)] + grid[(i-1) + s_nodes * (tau_reversed+1)]) / pow(dS, 2);
        double theta = -0.5 * pow(sigma, 2) * pow(current_s, 2) * gamma - rate * current_s * delta + rate * grid[i + s_nodes * (tau_reversed+1)];
        grid[i + s_nodes * tau_reversed] = grid[i + s_nodes * (tau_reversed+1)] - (theta * dt);


        // is this right?
        if(tau < t_nodes){
            if(i==0 || i==s_nodes){
                double lower, upper;
                std::tie(lower, upper) = set_boundary_conditions(tau, dt);;
                grid[0 + s_nodes * tau] = lower;
                grid[s_nodes + s_nodes * tau] = upper;
            }
        }
    }

}

// __device__ indicates that function is only used within the gpu
__device__ static std::tuple<double, double> set_boundary_conditions(int tau, double dt)
{
    double lower_boundary = 0;
    double upper_boundary = 0;

    double current_time_step = dt * tau;

    if (option_type == "call") {
        upper_boundary = s_max - strike_price * std::exp(-rate * (expiry - current_time_step));
    }else if (option_type == "put") {
        lower_boundary = strike_price * std::exp(-rate * (expiry - current_time_step));
    }

    return  std::make_tuple(lower_boundary, upper_boundary);
}

static void set_terminal_condition(double* grid, std::string option_type, int s_nodes, int t_nodes, double dS, double strike_price)
{
    if(option_type == "call"){
        for(int i=0; i<=s_nodes;i++){
            double current_s = i * dS;
            grid[i * t_nodes + t_nodes] = fmax(current_s - strike_price, 0);
        }
    }else if(option_type == "put"){
        for(int i=0; i<=s_nodes;i++){
            double current_s = i * dS;
            grid[i * t_nodes + t_nodes] = fmax(strike_price - current_s, 0);
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
    int s_nodes = 200;
    int t_nodes = 10000;

    double dt_max = 1 / (pow(s_nodes, 2) * pow(sigma, 2));
    double dS = s_max / s_nodes;
    double dt;

    // calculates appropriate dt value (is it okay to put this here?)
    if(t_nodes.has_value()){
        dt = expiry / t_nodes.value();
        assert(dt < dt_max);
    }else{
        dt = 0.9 * dt_max;
    }

    size_t grid_size = (s_nodes + 1) * (t_nodes + 1);

    // host memory allocation
    double* host_grid = (double*)malloc(grid_size * sizeof(double));

    // initiliasing host_grid values (is this allowed???)
    set_terminal_condition(host_grid, option_type, s_nodes, t_nodes, dS, strike_price);

    // gpu memory allocation
    double *dev_grid;
    cudaMalloc(&dev_grid, grid_size * sizeof(double)); // should i use this or cudamallocpitch ???

    // copy host memory to gpu memory
    cudaMemcpy(dev_grid, host_grid, grid_size * sizeof(float), cudaMemcpyHostToDevice);

    // kernel invocation
    solve_bse_explicit_parallel<<<numBlocks, threadsPerBlock>>>(dev_grid, option_type, s_max, expiry, sigma, rate, strike_price, s_nodes, t_nodes, dS, dt);

    cudaDeviceSynchronize();

    cudaMemcpy(host_grid, dev_grid,grid_size * sizeof(float), cudaMemcpyDeviceToHost);

}

