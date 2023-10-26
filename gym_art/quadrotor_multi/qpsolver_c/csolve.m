% csolve  Solves a custom quadratic program very rapidly.
%
% [vars, status] = csolve(params, settings)
%
% solves the convex optimization problem
%
%   minimize(0.5*quad_form(x, P) + q'*x)
%   subject to
%     G*x <= h
%     min(x) >= lb
%     max(x) <= ub
%
% with variables
%        x   3 x 1
%
% and parameters
%        G 140 x 3
%        P   3 x 3    PSD, diagonal
%        h 140 x 1
%       lb   1 x 1
%        q   3 x 1
%       ub   1 x 1
%
% Note:
%   - Check status.converged, which will be 1 if optimization succeeded.
%   - You don't have to specify settings if you don't want to.
%   - To hide output, use settings.verbose = 0.
%   - To change iterations, use settings.max_iters = 20.
%   - You may wish to compare with cvxsolve to check the solver is correct.
%
% Specify params.G, ..., params.ub, then run
%   [vars, status] = csolve(params, settings)
% Produced by CVXGEN, 2023-09-05 15:37:09 -0400.
% CVXGEN is Copyright (C) 2006-2017 Jacob Mattingley, jem@cvxgen.com.
% The code in this file is Copyright (C) 2006-2017 Jacob Mattingley.
% CVXGEN, or solvers produced by CVXGEN, cannot be used for commercial
% applications without prior written permission from Jacob Mattingley.

% Filename: csolve.m.
% Description: Help file for the Matlab solver interface.
