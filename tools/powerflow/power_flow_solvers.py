# M. Sinner 6/20/19
"""
Implement methods for solving the power flow equations. 

Currently only contains the Newton Raphson method (although this appears to
perform well).

Assumes a slack bus (with known voltage) and N `load' busses (with known
power), and solves for the load bus voltages (and slack bus power).

"""
import numpy as np

def Netwon_Raphson_method(admittance_matrix, s, v_init='Unity', v_0=1+0j, 
                          tolerance=1e-6, max_iterations=50, quiet=False):
    """
    Use Newton-Raphson method to solve power flow problem.
    
    Inputs:
        admittance_matrix - (N+1)x(N+1) array of complex - system 
                                                           admittance
                                                           matrix
        s - Nx1 array of complex - nodal complex powers at `load' nodes
        v_init - Nx1 array of complex - initial guess of nodal complex
                                        voltages at `load' nodes
        v_0 - complex - voltage at the slack bus (defaults to per unit)
        tolerance - float - solution tolerance
        max_iterations - int or float - maximum allowable number of 
                                        iterations
        quiet - Boolean - silence printed outputs
    
    Outputs:
        s - (N+1)x1 array of complex - nodal complex powers
        i - (N+1)x1 array of complex - nodal complex currents
        v - (N+1)x1 array of complex - nodal complex voltages
    """

    if v_init is 'Unity':
        v_init = np.ones(np.shape(s))

    N = int(round(np.shape(s)[0]))

    s_prime = np.concatenate((np.real(s), np.imag(s)))

    # construct the matrices of the real form
    Y_LL_prime = _build_Y_prime(admittance_matrix[1:,1:])
    Y_L0_prime = _build_Y_prime(admittance_matrix[1:,0].reshape(-1,1))

    # Initialize
    v_prime = np.concatenate((np.real(v_init), np.imag(v_init)))
    v_0_prime = np.concatenate((np.reshape(np.real(v_0), [1,1]),
                                np.reshape(np.imag(v_0), [1,1])))
    
    # Run iterative algorithm
    for iteration in range(max_iterations):
        f_v = power_flow_mismatch_function(Y_LL_prime,
                                           Y_L0_prime,
                                           v_prime, 
                                           v_0_prime,
                                           s_prime)
        
        Jac_f_v = power_flow_Jacobian(Y_LL_prime,
                                      Y_L0_prime,
                                      v_prime,
                                      v_0_prime)

        correction = -np.linalg.solve(Jac_f_v, f_v)

        if (abs(f_v) <= tolerance).all():
            if not quiet:
                print('Solution satisfying tolerance found using ' + \
                      'Newton-Raphson method after %d iterations.'
                      % (iteration+1))
            break
        elif (abs(f_v) >= 10e10).all():
            print('Solution has diverged. Terminating execution.')
            break
        elif iteration == max_iterations - 1:
            if not quiet:
                print('Maximum number of iterations (%d)'% (max_iterations) + \
                       ' reached before a satisfactory solution found.')
        else:
            v_prime = v_prime + correction

    solution = power_flow_mismatch_function(Y_LL_prime,
                                            Y_L0_prime,
                                            v_prime, 
                                            v_0_prime,
                                            s_prime)
    if not quiet:                                            
        print('Solution mismatch:')
        print((solution[:N] + solution[N:]*1j))
    
    s_prime_solved = solution + s_prime
    s = s_prime_solved[:N] + s_prime_solved[N:]*1j
    v = v_prime[:N] + v_prime[N:]*1j
    v_all = np.concatenate((np.reshape(v_0, [1,1]), v))
    i_all = admittance_matrix @ v_all
    s_all = np.diag(v_all.T[0]) @ np.conj(i_all)
    
    return s_all, v_all, i_all

def power_flow_mismatch_function(Y_LL_prime, Y_L0_prime, v_prime, v_0_prime, 
                                 s_prime):
    """
    Calculate output of power flow mismatch equation.

    Inputs:
        Y_LL_prime - 2Nx2N array of real - load bus admittance matrix
        Y_L0_prime - 2Nx2 array of real - slack to load bus admittance 
        v_prime - 2Nx1 array of real - load voltages
        v_0_prime - 2x1 array of real - slack voltages
        s_prime - 2Nx1 array of real - load bus powers
    Outputs:
        (mismatch) - 2Nx1 array of real - power flow mismatches at load
                                          busses
    """
    N = int(round(np.shape(s_prime)[0]/2))
    v_matrix = np.append(np.append(np.diag(v_prime[:N].T[0]), 
                                   np.diag(v_prime[N:].T[0]),
                                   1),
                         np.append(np.diag(v_prime[N:].T[0]),
                                   -np.diag(v_prime[:N].T[0]),
                                   1),
                         0)    

    return s_prime - v_matrix @ (Y_LL_prime@v_prime + Y_L0_prime@v_0_prime)

def power_flow_Jacobian(Y_LL_prime, Y_L0_prime, v_prime, v_0_prime):
    """
    Find Jacobian (w.r.t v') of power flow mismatch equation.

    Inputs:
        Y_LL_prime - 2Nx2N array of real - load bus admittance matrix
        Y_L0_prime - 2Nx2 array of real - slack to load bus admittance 
        v_prime - 2Nx1 array of real - load voltages
        v_0_prime - 2x1 array of real - slack voltages
    
    Outputs:
        Jac - 2Nx2N array of real - system Jacobian matrix
    """
    N = int(round(np.shape(Y_LL_prime)[0]/2))
    Jac = np.zeros([2*N, 2*N])
    for k in range(N):
        for m in range(N):
            if m == k:
                kronecker_delta = 1
            else:
                kronecker_delta = 0
            
            # Top left (R,R)
            Jac[k,m] =  - (v_prime[k][0] * Y_LL_prime[k,m] + \
                           v_prime[N+k][0] * Y_LL_prime[N+k,m]) \
                        - kronecker_delta * (Y_LL_prime[k,:] @ v_prime + \
                                             Y_L0_prime[k,:] @ v_0_prime)
                       
            
            # Top right (R,I)
            Jac[k,N+m] = - (v_prime[N+k][0] * Y_LL_prime[N+k,N+m] + \
                            v_prime[k][0] * Y_LL_prime[k,N+m]) \
                          - kronecker_delta * (Y_LL_prime[N+k,:] @ v_prime + \
                                               Y_L0_prime[N+k,:] @ v_0_prime)
                          
            
            # Bottomr left (I,R)
            Jac[N+k,m] = - (v_prime[N+k][0] * Y_LL_prime[k,m] - \
                            v_prime[k][0] * Y_LL_prime[N+k,m]) \
                          + kronecker_delta * (Y_LL_prime[N+k,:] @ v_prime + \
                                               Y_L0_prime[N+k,:] @ v_0_prime)
                          
            
            # Bottom right (I,I)
            Jac[N+k,N+m] = - (v_prime[N+k][0] * Y_LL_prime[k,N+m] - \
                              v_prime[k][0] * Y_LL_prime[N+k,N+m]) \
                           - kronecker_delta * (Y_LL_prime[k,:] @ v_prime +
                                                Y_L0_prime[k,:] @ v_0_prime)

    return Jac


def _build_Y_prime(admittance_matrix):
    """ 
    Construct the admittance matrix for the real form.

    Inputs:
        admittance_matrix - (N+1)x(N+1) array of complex - system 
                                                           admittance
                                                           matrix
        
    Outputs:
        Y_prime - (2(N+1) x 2(N+1)) array - real form of admittance 
                                            matrix
    """
    Y_prime = np.append(np.append(np.real(admittance_matrix), 
                                  -np.imag(admittance_matrix), 
                                  1),
                        np.append(np.imag(admittance_matrix),
                                  np.real(admittance_matrix),
                                  1),
                        0)
    return Y_prime


# def Z_bus_method(admittance_matrix, s, v0='Unity', tolerance=1e-6,
#                                max_iterations=50, quiet=False):
#     """
#     Solve power flow problem in Z-bus form.
#     Based on Wang, Bernstein, Le Boudec, & Paolone (2018)
#     Employs a fixed point iteration method. Assumes that the first node
#     in the network is a slack bus, while all others are PQ-busses.

#     Assumes calculations will be carried out on a per-unit basis.
    
#     Inputs:
#         admittance_matrix - (N+1)x(N+1) array of complex - system 
#                                                            admittance
#                                                            matrix
#         s - Nx1 array of complex - nodal complex powers at PQ buses
#         v0 - Nx1 array of complex - initial guess of nodal complex
#                                     voltages
#         tolerance - float - solution tolerance
#         max_iterations - int or float - maximum allowable number of 
#                                         iterations
#         quiet - Boolean - silence printed outputs
    
#     Outputs:
#         s - (N+1)x1 array of complex - nodel complex powers at slack and
#                                        PQ busses
#         i - (N+1)x1 array of complex - nodal complex currents
#         v - (N+1)x1 array of complex - nodal complex voltages
#     """

#     # Notation is based on Wang, Bernstein, Le Boudec, & Paolone (2018)
#     N = np.shape(admittance_matrix)[0] - 1
#     Y_00 = admittance_matrix[0,0]
#     Y_0L = admittance_matrix[0,1:].reshape(1,-1)
#     Y_LL = admittance_matrix[1:,1:]
#     Y_L0 = admittance_matrix[1:,0].reshape(-1,1)
    
#     zero_load_voltage = -np.linalg.solve(Y_LL, Y_L0)
    
#     v_k = v0 # Initialize
#     for iteration in range(max_iterations):
        
#         # import ipdb; ipdb.set_trace()
#         v_kp1 = _Z_bus_flow(v_k, zero_load_voltage, Y_LL, s) # new estimate

#         if (abs(v_kp1 - v_k) <= tolerance).all():
#             if not quiet:
#                 print('Solution satisfying tolerance found using ' + \
#                       'Z-bus method after %d iterations.'
#                       % (iteration+1))
#             break
#         elif (abs(v_kp1 - v_k) >= 1e3).any():
#             if not quiet:
#                 print('Solution has diverged. Terminating execution.')
#             break
#         elif iteration == max_iterations - 1:
#             if not quiet:
#                 print('Maximum number of iterations (%d)'% (max_iterations) + \
#                        ' reached before a satisfactory solution found.')
#         else:
#             v_k = v_kp1 # update estimate

#     # Build full solution including slack bus
#     v = np.append(np.array([[1+0j]]), v_kp1, axis=0)
#     i = admittance_matrix @ v
#     s = np.diag(v[:,0]) @ np.conj(i)
#     return s, v, i


# def _Z_bus_flow(v, w, Y_LL, s):
#     """"
#     Perform step of Z_bus_method fixed point iterative algorithm.
    
#     Inputs:
#         v - Nx1 array of complex - voltages of N nodes to be iterated
#         w - Nx1 array of complex - zero load voltages
#         Y_LL - NxN array of complex - admittance matrix for network with 
#                                       slack bus removed
#         s - Nx1 array of complex - power injections at N nodes
#     """
#     #import ipdb; ipdb.set_trace()
#     x = np.linalg.solve(np.diag(np.conj(v)[:,0]), np.conj(s))

#     return w + np.linalg.solve(Y_LL, x)

    




# # FUNCTIONS BELOW THIS LINE ARE CURRENTLY UNUSED

# def power_flow_function_k(Hessian_k, v, s):
#     """
#     Based on the Hessian
#     """
#     return s - v.T @ Hessian_k @ v


# def power_flow_gradient_k(Hessian_k, v):
#     """ 
#     Each column of the output is the gradient for a single node;
    
#     grad[:][0] pertains to s_1 - ...,
#     grad[:][1] pertains to s_2 - ...,
#     :
#     etc
#     :
#     grad[:][N-1] pertains to s_N - ....

#     Jacobian? Or Jacobian transposed, perhaps??
#     """ 

#     return Hessian_k @ v

# def power_flow_Hessian(Y):
#     """ 
#     3D array. DO I EVEN NEED THE HESSIAN FOR ANYTHING?
    
#     Hessian[:][:][0] pertains to s_1 - ...,
#     Hessian[:][:][1] pertains to s_1 - ...,
#     :
#     etc
#     :
#     Hessian[:][:][N-1] pertains to s_N - ....
#     """
#     N = np.shape(Y)[0]
#     Hessian = np.zeros([2*N, 2*N, 2*N]) # Contains real and imaginary parts
    
#     for k in range(N):
#         H_kR = np.zeros([2*N, 2*N])
#         H_kI = np.zeros([2*N, 2*N])

#         for m in range(N):
#             H_kR[:2*k+2][2*m:2*m+2] = np.array([[np.real(Y[k][m]), 
#                                                      -np.imag(Y[k][m])],
#                                                     [np.imag(Y[k][m]),
#                                                      np.real(Y[k][m])]])
            
#             H_kI[:2*k+2][2*m:2*m+2] = np.array([[-np.imag(Y[k][m]), 
#                                                      -np.real(Y[k][m])],
#                                                     [np.real(Y[k][m]),
#                                                      -np.imag(Y[k][m])]])

#         Hessian[:,:,2*k:2*k+2] = H_kR + H_kR.T
#         Hessian[:,:,2*k+2:2*k+4] = H_kI + H_kI.T
    
#     return Hessian
        

        
    
