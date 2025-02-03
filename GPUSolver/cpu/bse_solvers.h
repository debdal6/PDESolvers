//
// Created by Chelsea De Marseilla on 01/02/2025.
//

#ifndef BSE_EXPLICIT_H
#define BSE_EXPLICIT_H

#include <string>
#include <vector>
#include <optional>

class BSE {
public:
    // Public members
    std::string option_type;
    double s_max;
    int expiry;
    double sigma;
    double rate;
    double strike_price;
    int s_nodes;
    std::optional<int> t_nodes;

    // Constructor
    BSE(std::string option_type, double s_max, int expiry, double sigma, double r, double k, int s_nodes, std::optional<int> t_nodes = std::nullopt);

    // Public method
    void solve_bse_explicit(std::vector<std::vector<double>>& grid);
    void solve_bse_cn(std::vector<std::vector<double>>& grid);

private:
    // Private method
    std::tuple<double, double> set_boundary_conditions(int tau, double dt);
};

#endif //BSE_EXPLICIT_H
