from utils.convex_min_flow import ConvexMinFlow
import numpy as np

#TODO use the solver for this
def solve_lp(function_value, gradient):
    return 0


def solve_frank_wolfe_convex_min_flow(cmf: ConvexMinFlow, max_iter : int = 1000, eps : np.float64 = 1e-4, verbose:bool = False):
    """
    The function solves the Convex Min Flow problem on the provided instance cmf using the Frank-Wolfe method.

    Args:
        cmf (ConvexMinFlow): The problem instance.
        max_iter (int): The maximum number of possible iterations of the algorithm.
        eps (np.float.64): The precision of the solution.
        verbose (bool): Flag that outpus extra information during the execution.
    
    Returns:
        tuple[np.float64, np.array]: The best f(x) found and the respective x.
    """
    x = cmf.u / 2 # starts from the middle of the box
    best_lower_bound = -np.inf

    for i in range(0,max_iter):
        # Compute f(x)= 1/2 \sum_i d_i*x_i^2 + \sum_j q_j*x_j and direction = -gradient
        f_x = 0.5 * (cmf.Q * x * x) + cmf.q * x
        g = - (cmf.Q * x + cmf.q)

        # Solve LP problem
        y = solve_lp(f_x,g)

        # Compute lower bound
        lower_bound = f_x + g*(y-x)

        if lower_bound > best_lower_bound:
            best_lower_bound = lower_bound

        # Compute the relative gap
        rel_gap = (f_x-best_lower_bound) / max(np.asb(f_x),np.ones_like(f_x))

        # Output statistics
        if verbose:
            print(f"i:{i}, f(x){f_x}, best lower bound so far: {best_lower_bound}, relative gap:{rel_gap}")

        # Stopping criterion
        if rel_gap <= eps:
            status = 'optimal'
            break

        # TODO: In stabilized case, restrict y in the box
        # restrict y

        # Compute direction
        d = y-x
        # Compute optimal unbounded step-size (in theory Line search for alpha)
        # compute optimal unbounded stepsize:
        # min (1/2) ( x + a d )' * Q * ( x + a d ) + q' * ( x + a d ) =
        #     (1/2) a^2 ( d' * Q * d ) + a d' * ( Q * x + q ) [ + const ]
        #
        # ==> a = - d' * ( Q * x + q ) / d' * Q * d
        den = d * cmf.Q * d
        if den < 1e-16: # d * Q * d = 0 => f is linear along d, take the max step size possible
            alpha = 1
        else:
            alpha = min((g*d)/den, 1)
        
        # Compute new point
        x = x + alpha * d
        if i == max_iter-1:
            status = 'stopped'
    
    print(f"Stop at iter: {i}, status: {status}, best f(x):{f_x}, relative gap: {rel_gap}")
    return (f_x, x)

        

if __name__ == '__main__':
    x = np.array([[2,2,2]])
    q = np.array([[1,1,1]])
    print(q*x*x)
