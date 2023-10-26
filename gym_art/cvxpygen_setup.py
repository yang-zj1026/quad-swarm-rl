import cvxpy as cp
import numpy as np

# Generate a random non-trivial quadratic program.
m = 80
n = 3
P = cp.Parameter((n, n), name='P', PSD=True)
q = cp.Parameter(n, name='q')
G = cp.Parameter((m, n), name='G')
h = cp.Parameter(m, name='h')

P.value = 2 * np.eye(n)
q.value = np.random.randn(n)
G.value = np.random.randn(m, n)
h.value = np.random.randn(m)

# Define and solve the CVXPY problem.
x = cp.Variable(n, name='x')
quad_form = cp.sum_squares(P @ x)

prob = cp.Problem(cp.Minimize((1 / 2) * quad_form + q.T @ x),
                  [G @ x <= h, 0 <=x, x <= 100])
prob.solve()

from cvxpygen import cpg

cpg.generate_code(prob, code_dir='qpsolver', solver='OSQP')

# import time
# import sys
#
# # import extension module and register custom CVXPY solve method
# from qpsolver.cpg_solver import cpg_solve
# prob.register_solve('cpg', cpg_solve)
#
# # solve problem conventionally
# t0 = time.time()
# val = prob.solve(solver='OSQP')
# t1 = time.time()
# sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
# print("Primal Solution", x.value)
# print("Dual Solution", prob.constraints[0].dual_value)
# print('Problem status: %s' % prob.status)
# sys.stdout.write('Objective function value: %.6f\n' % val)
#
# # solve problem with C code via python wrapper
# t0 = time.time()
# val = prob.solve(method='cpg', updated_params=['P', 'q', 'G', 'h'])
# t1 = time.time()
# sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
# print("Primal Solution", x.value)
# print("Dual Solution", prob.constraints[0].dual_value)
# print('Problem status: %s' % prob.status)
# sys.stdout.write('Objective function value: %.6f\n' % val)
