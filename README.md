### MVDONE ###

MVDONE uses a piece-wise linear surrogate model for optimization of expensive cost functions with continuous and discrete variables.

`MVDONE_minimize(obj, x0, lb, ub, num_int, max_evals, rand_evals)` solves the minimization problem

**min** *f(x)*

**st.** *lb<=x<=ub, the first num_int variables of x are integer*

where `obj` is the objective function, `x0` the initial guess,
`lb` and `ub` are the bounds, `num_int` is the number of integer variables,
and `max_evals` is the maximum number of objective evaluations (`rand_evals` of these
are random evaluations).

It is the mixed-variable version of the [DONE algorithm](https://bitbucket.org/csi-dcsc/donecpp/src/master/), 
meant for problems with both continuous and discrete variables.

Laurens Bliek, 06-03-2019

Dependencies:

* numpy
* scipy
* matplotlib
* time
* For comparison with other methods: hyperopt, gpy, pandas

Please contact l.bliek@tudelft.nl if you have any questions.