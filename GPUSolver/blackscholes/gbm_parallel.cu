//
// Created by Chelsea De Marseilla on 03/01/2025.
//

#include <cmath>
#include <iostream>
#include <curand_kernel.h>

#define SEED 12345

__global__ static void simulate_gbm(float* grid, float* brownian_path, float initial_stock_price, float mu, float sigma, float time, int time_steps, int num_of_simulations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_of_simulations)
    {

        // initialising for random normal distribution
        curandState state;
        curand_init(SEED, idx, 0, &state);

        float dt = time/static_cast<float>(time_steps);

        brownian_path[idx * time_steps] = 0.0f;
        grid[idx * time_steps] = initial_stock_price;

        for (int i = 1; i < time_steps; i++)
        {
            float Z = curand_normal(&state);
            brownian_path[idx * time_steps + i] = brownian_path[idx * time_steps + i - 1] + std::sqrt(dt) * Z;
            grid[idx * time_steps + i] = grid[idx * time_steps + i - 1] * expf((mu - 0.5f * powf(sigma, 2)) * dt + sigma * (brownian_path[idx * time_steps + i] - brownian_path[idx * time_steps + i - 1]));
        }
    }
}

int main()
{

    float initial_stock_price = 100.0f;
    float mu = 0.05f;
    float sigma = 0.03f;
    float time = 1;
    int time_steps = 365;
    int num_of_simulations = 100;

    int block_size = 256;
    int num_blocks = (num_of_simulations + block_size - 1) / block_size;

    size_t grid_size = num_of_simulations * time_steps;

    // host memory allocation
    float* host_grid = (float*)malloc(grid_size * sizeof(float));

    // gpu memory allocation
    float *dev_grid, *bm;
    cudaMalloc(&dev_grid, grid_size * sizeof(float));
    cudaMalloc(&bm, num_of_simulations * time_steps * sizeof(float));

    // copy host memory to gpu memory
    cudaMemcpy(dev_grid, host_grid, grid_size * sizeof(float), cudaMemcpyHostToDevice);

    // kernel invocation
    simulate_gbm<<<num_blocks, block_size>>>(dev_grid, bm, initial_stock_price, mu, sigma, time, time_steps, num_of_simulations);

    // waits kernel to finish all processes
    cudaDeviceSynchronize();

    // copy updated gpu memory to host memory
    cudaMemcpy(host_grid, dev_grid,grid_size * sizeof(float), cudaMemcpyDeviceToHost);


    // printing for debugging
    for (int i = 0; i < 10; i++)
    {
        std::cout << "Simulation " << i << ": ";
        for (int j = 0; j < time_steps; j++)
        {
            std::cout << host_grid[i * time_steps + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(dev_grid);
    cudaFree(bm);

    free(host_grid);
}
