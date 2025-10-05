from typing import Callable, Any
import numpy as np

def bisect(f, a, b, eps, n_max, store_midpoints=True):
    """
    Bisection method for finding a root of a continuous function on [a, b].

    Parameters
    ----------
    Same as before .. 
    store_midpoints : bool, optional
        If True, also return the list of midpoints at each iteration.

    Returns
    -------
    Same as before .. 
    midpoints : List[float], optional
        List of midpoints (if store_midpoints=True).
    """
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")

    a_new, b_new = a, b
    x = np.mean([a, b])
    err = 1 + eps
    errors = [err]
    it = 0

    midpoints = [x] if store_midpoints else []

    while err > eps and it < n_max:
        if f(a_new) * f(x) < 0:
            b_new = x
        else:
            a_new = x

        x_new = np.mean([a_new, b_new])
        err = abs(x_new - x)
        errors.append(err)

        x = x_new
        if store_midpoints:
            midpoints.append(x)
        it += 1

    if store_midpoints:
        return x, it, errors, midpoints
    else:
        return x, it, errors
    
class Solver():
    """
    Collection of methods for solving equations

    Attributes
    ----------
    f: Callable
        A function f: R -> R whose roots are to be found.
    df: Callable
        The derivative of f.
    
    Methods
    -------
        bisection
        newton
        chord
        secant
    
    """
    def __init__(self, f: Callable[[float], float], df: Callable | None = None):
        self.f = f
        self.df = df

    def bisection(self, 
                  a: float, 
                  b: float, 
                  eps: float = 1e-9, 
                  max_iter: float = 50,
                  history: bool = True) -> dict[str, Any]:

        solution = {
            'root': None,
            'eps': eps,
            'max_iter': max_iter,
            'errors': [],
            'midpoints': [],
            'n_iter': 0
        }
        midpoints = []
        if history:
            root, n_iter, errors, midpoints = bisect(self.f, a, b, eps, max_iter, history)
        else:
            root, n_iter, errors = bisect(self.f, a, b, eps, max_iter, history)
        
        solution['root'] = root  
        solution['n_iter'] = n_iter 
        solution['errors'] = errors
        solution['midpoints'] = midpoints 

        return solution

    def newton(self, 
                x0: float, 
                eps: float = 1e-9, 
                max_iter: float = 50,
                history: bool = True) -> dict[str, any]: 
        
        solution = {
            'root': None,
            'eps': eps,
            'max_iter': max_iter,
            'errors': [],
            'midpoints': [],
            'n_iter': 0
        }  
        midpoints = []
        if np.abs(self.df(x0)) < 1e-16:
            raise ValueError("Derivative at initial guess is too small.")

        # Initialization
        err = float("inf")
        errors = [err]
        it = 0
        x = x0
        solution['midpoints'] = [x] if history else []
        # Iteration
        while err > eps and it < max_iter:
            qk = self.df(x)

            if abs(qk) < 1e-12:
                raise RuntimeError("Derivative too close to zero during iteration.")

            # Newton update
            x_new = x - self.f(x) / qk

            # Error estimate (difference in successive iterates)
            err = abs(x_new - x)
            errors.append(err)

            # Update for next iteration
            x = x_new
            if history:
                solution['midpoints'].append(x)

            it += 1 
        
        solution['root'] = x
        solution['errors'] = errors
        solution['n_iter'] = it

        return solution
    
    def chord(self, a, b, x0, eps, max_iter, history = True):
        """
        Implementation of the chord (secant) method to find a root of f(x).
        
        Parameters
        ----------
            f      : Function whose root is sought.
            a, b   : Interval endpoints used to calculate initial slope.
            x0     : Initial guess.
            eps    : Desired accuracy (default 1e-6).
            n_max  : Maximum number of iterations (default 100).
        
        Returns
        -------
            A dict with found root, error, n_iter and fixed parameters
        """

        solution = {
            'root': None,
            'eps': eps,
            'max_iter': max_iter,
            'errors': [],
            'midpoints': [],
            'n_iter': 0
        } 

        q = (self.f(b) - self.f(a)) / (b - a)
        errors = [eps + 1.]  # Initialize with a large error
        x = x0
        solution['midpoints'] = [x] if history else []
        it = 0

        while errors[-1] > eps and it < max_iter:
            x_new = x - self.f(x) / q
            err = abs(x_new - x)
            errors.append(err)
            x = x_new
            if history:
                solution['midpoints'].append(x)

            it += 1

        solution['root'] = x
        solution['errors'] = errors
        solution['n_iter'] = it


        return solution
    
    def secant(self, x0, x00, eps, max_iter, history=True):
        """
        Implementation of the secant method to find a root of a function f(x).

        Parameters:
            f      : callable
                    The function whose root is to be found.
            x0     : float
                    The first initial guess for the root.
            x00    : float
                    The second initial guess for the root.
            eps    : float
                    Desired accuracy; the iteration stops when the absolute change 
                    between consecutive approximations is less than eps.
            n_max  : int
                    Maximum number of iterations allowed.

        Returns:
            xk     : float
                    The approximated root of the function.
            it     : int
                    The number of iterations performed.
            errors : list of float
                    List of absolute errors at each iteration 
                    (|x_new - x_old|).
        """

        solution = {
            'root': None,
            'eps': eps,
            'max_iter': max_iter,
            'errors': [],
            'midpoints': [],
            'n_iter': 0
        } 

        err = eps + 1.
        errors = [err]
        it = 0
        xk = x0
        xkk = x00
        solution['midpoints'] = [xk] if history else []

        while (err > eps and it < max_iter):
            qk = (self.f(xk) - self.f(xkk))/(xk - xkk)
            x_new = xk - self.f(xk)/qk
            err = abs(x_new - xk)
            xkk = xk
            xk = x_new
            if history:
                solution['midpoints'].append(xk)
            errors.append(err)
            it += 1
        solution['root'] = xk
        solution['errors'] = errors
        solution['n_iter'] = it

        return solution
