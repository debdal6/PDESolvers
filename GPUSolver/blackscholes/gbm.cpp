//
// Created by Chelsea De Marseilla on 06/01/2025.
//

#include <iostream>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>

#define SEED 12345

static void simulate_gbm(float initial_stock_price, float mu, float sigma, float time, int time_steps, int num_of_simulations, std::vector<std::vector<float>> &grid, std::vector<std::vector<float>> &bm)
{

    // std::vector<std::vector<float>> grid(num_of_simulations, std::vector<float>(time_steps, 0.0));
    // std::vector<std::vector<float>> bm(num_of_simulations, std::vector<float>(time_steps, 0.0));

    std::mt19937 gen(SEED);
    std::normal_distribution<float> distribution(0.0f, 1.0);

    float dt = time/static_cast<float>(time_steps);

    // setting initial price for all simulations
    for (int i=0; i<num_of_simulations; i++)
    {
        grid[i][0] = initial_stock_price;
    }

    for (int i=0; i<num_of_simulations; i++)
    {
        for (int j=1; j<time_steps; j++)
        {
            float Z = distribution(gen);
            bm[i][j] = bm[i][j-1] + std::sqrt(dt) * Z;
            grid[i][j] = grid[i][j - 1] * expf((mu - 0.5f * powf(sigma, 2)) * dt + sigma * (bm[i][j] - bm[i][j - 1]));
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

    std::vector<std::vector<float>> grid(num_of_simulations, std::vector<float>(time_steps, 0.0));
    std::vector<std::vector<float>> bm(num_of_simulations, std::vector<float>(time_steps, 0.0));

    auto start = std::chrono::high_resolution_clock::now();
    simulate_gbm(initial_stock_price, mu, sigma, time, time_steps, num_of_simulations, grid, bm);
    auto end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10; i++)
    {
        std::cout << "Simulation " << i << ": ";
        for (int j = 0; j < time_steps; j++)
        {
            std::cout << grid[i][j] << " ";
        }
        std::cout << std::endl;
    }

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Duration of CPU Performance: " << duration.count() << " microseconds" << std::endl;


    // std::ofstream output_file("/home/chemardes/simulation_results_serial.csv");
    //
    // // Write header (optional)
    // output_file << "Simulation,Time Step,Stock Price\n";
    //
    // // Output simulation results to CSV (only stock price)
    // for (int i = 0; i < num_of_simulations; i++)
    // {
    //     for (int j = 0; j < time_steps; j++)
    //     {
    //         output_file << i << "," << j << "," << grid[i][j] << "\n";
    //     }
    // }
    //
    // // Close the file
    // output_file.close();

}

