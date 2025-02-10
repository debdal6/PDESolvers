//
// Created by Chelsea De Marseilla on 01/02/2025.
//

#include <iostream>
#include <optional>
#include <string>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <vector>
#include <tuple>

#include "bse_solvers.h"

#include <chrono>
#include <fstream>

BSE::BSE(std::string option_type, double s_max, int expiry, double sigma, double r, double k, int s_nodes, std::optional<int> t_nodes)
    : option_type(option_type), s_max(s_max), expiry(expiry), sigma(sigma), rate(r), strike_price(k), s_nodes(s_nodes), t_nodes(t_nodes) {}

void BSE::solve_bse_explicit(std::vector<std::vector<double>> &grid){

    double dt_max = 1 / (pow(s_nodes, 2) * pow(sigma, 2));
    double dS = s_max / s_nodes;
    double dt;

    if(t_nodes.has_value()){
        dt = static_cast<double>(expiry) / t_nodes.value();
        assert(dt < dt_max);
    }else{
        dt = 0.9 * dt_max;
    }

    // setting terminal condition for each option type
    if(option_type == "call"){
        for(int i=0; i<=s_nodes;i++){
            double current_s = i * dS;
            grid[i][t_nodes.value()] = fmax(current_s - strike_price, 0);
        }
    }else if(option_type == "put"){
        for(int i=0; i<=s_nodes;i++){
            double current_s = i * dS;
            grid[i][t_nodes.value()] = fmax(strike_price - current_s, 0);
        }
    }

    // computing price of option
    for(int tau=t_nodes.value()-1; tau>=0;tau--){
        for(int i=1; i<s_nodes; i++){
            double current_s = i * dS;

            double delta = (grid[i+1][tau+1] - grid[i-1][tau+1]) /  (2 * dS);
            double gamma = (grid[i+1][tau+1] - 2 * grid[i][tau+1] + grid[i-1][tau+1]) / pow(dS, 2);
            double theta = -0.5 * pow(sigma, 2) * pow(current_s, 2) * gamma - rate * current_s * delta + rate * grid[i][tau+1];
            grid[i][tau] = grid[i][tau + 1] - (theta * dt);

        }

        // setting boundary conditions
        double lower, upper;
        std::tie(lower, upper) = set_boundary_conditions(tau, dt);;
        grid[0][tau] = lower;
        grid[s_nodes][tau] = upper;
    }

}

std::tuple<double, double> BSE::set_boundary_conditions(int tau, double dt) {
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


int main()
{
    std::string option_type = "call";
    double s_max = 300.0;
    double expiry = 0.1;
    double sigma = 0.2;
    double rate = 0.05;
    double strike_price = 100;
    int s_nodes = 100;
    int t_nodes = 100000;

    BSE bse(option_type, s_max, expiry, sigma, rate, strike_price, s_nodes, t_nodes);

    std::vector<std::vector<double>> grid_explicit(s_nodes+1, std::vector<double>(t_nodes+1, 0.0));

    auto start = std::chrono::high_resolution_clock::now();

    bse.solve_bse_explicit(grid_explicit);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Duration of CPU Performance: " << (double) duration.count() / 1e6 << "s" << std::endl;

}
