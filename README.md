# MVRSM #

MVRSM uses a piece-wise linear surrogate model for optimisation of expensive cost functions with continuous and discrete variables.

`MVRSM_minimize(obj, x0, lb, ub, num_int, max_evals, rand_evals)` solves the minimisation problem

**min** *f(x)*

**st.** *lb<=x<=ub, the first num_int variables of x are integer*

where `obj` is the objective function, `x0` the initial guess,
`lb` and `ub` are the bounds, `num_int` is the number of integer variables,
and `max_evals` is the maximum number of objective evaluations (`rand_evals` of these
are random evaluations).

It is the mixed-variable version that is related to the [DONE algorithm](https://bitbucket.org/csi-dcsc/donecpp/src/master/), 
meant for problems with both continuous and discrete variables.

Laurens Bliek, 06-03-2020

## How to cite ##

```bibtex
@inproceedings{10.1145/3449726.3463136,
author = {Bliek, Laurens and Guijt, Arthur and Verwer, Sicco and de Weerdt, Mathijs},
title = {Black-box mixed-variable optimisation using a surrogate model that satisfies integer constraints},
year = {2021},
isbn = {9781450383516},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3449726.3463136},
doi = {10.1145/3449726.3463136},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
pages = {1851–1859},
numpages = {9},
location = {Lille, France},
series = {GECCO '21}
}
```

## Dependencies ##

* numpy
* scipy
* matplotlib
* time
* For comparison with other methods: hyperopt, gpy, pandas


## How to run ##

Run the file `demo.py` directly, or indicate an existing test function to optimise, for example:

`python demo.py -f dim10Rosenbrock  -n 10 -tl 4`

Here, `-f` is the function to be optimised, `-n` is the number of iterations, and `-tl` is the total number of runs.
Currently the following test functions are already supported:
'func2C', 'func3C', 'dim10Rosenbrock', 'linearmivabo', 'dim53Rosenbrock', 'dim53Ackley', 'dim238Rosenbrock'

Afterward, use `plot_result.py` for visualisation (set the correct folder and other parameters inside the file).

Please contact l.bliek@tue.nl if you have any questions.
