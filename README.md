# alternating-direction-method-of-multiplier

## Reference
Augmented Lagrangian Method
- https://en.wikipedia.org/wiki/Augmented_Lagrangian_method

ADMM algorithm `(KOR)`
- https://kr.mathworks.com/help/stats/lasso.html#bvm6lzz_head

## Alternating Direction Method of Multipliers (ADMM)
Augmented Lagrangian methods are a certain class of algorithms for solving constrained optimization problems. They have similarities to [penalty methods](https://en.wikipedia.org/wiki/Penalty_method) in that they replace a constrained optimization problem by a series of unconstrained problems and add a penalty term to the objective; the difference is that the augmented Lagrangian method adds yet another term, designed to mimic a [Lagrange multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier). The augmented Lagrangian is not the same as the method of Lagrange multipliers.

Specially, The alternating direction method of multipliers (ADMM) is a variant of the augmented Lagrangian scheme that uses partial updates for the dual variables. 

## Cost function 
Cost function is fomulated by data fidelty term `f(x)` and regularization term `g(x)` as follow,

        (P1) argmin_x f(x) + g(x)

(P1) is equivalent to the constrained problem,

        (P2) argmin_x,y f(x) + g(y)
        
             subject to x = y
        
By the augmented lagrangian method, (P2) can be formulated to the unconstrained problem,

        L(x, y, u) = f(x) + g(y) + 1/2 * || x - y - u ||_2^2.
        
Then, the cost function `L` is separable in `x` and `y`.

`The subproblem of x` is as follow,

        x_(k+1) = argmin_x L(x, y_(k), u_(k))
        
                = argmin_x f(x) + 1/2 * || x - y_(k) - u_(k) ||_2^2.

`The subproblem of y` is fomulated as follow,

        y_(k+1) = argmin_y L(x_(k+1), y, u_(k))
        
                = argmin_y g(y) + 1/2 * || x_(k+1) - y - u_(k) ||_2^2.
                
The `u` is directly updated,

        u_(k+1) = x_(k+1) - y_(k+1) - u_(k).
