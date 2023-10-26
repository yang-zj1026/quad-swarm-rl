// Load and instantiate global namespaces with relevant data and working space.
#include "solver.h"
Vars vars;
Params params;
Workspace work;
Settings settings;

void load_data(Params params){
    // Load data into params
    params.P = ...;
    params.q = ...;
    params.G = ...;
    params.h = ...;
    params.lb = ...;
    params.ub = ...;
}

void use_solution(Vars vars){
    // Use solution
}

int main(int argc, char **argv) {
  set_defaults();  // Set basic algorithm parameters.
  settings.verbose = 0;  // Disable output.
//  setup_indexing();

  for (;;) {  // Main control loop.
    load_data(params);

    // Solve our problem at high speed!
    num_iters = solve();
    // Recommended: check work.converged == 1.

    use_solution(vars)
  }
}