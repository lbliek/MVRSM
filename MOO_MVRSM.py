# MVRSM uses a piece-wise linear surrogate model for optimization of
# expensive cost functions with mixed-integer variables.
#
# MVRSM_minimize(obj, x0, lb, ub, num_int, max_evals, rand_evals) solves the minimization problem
#
# min f(x)
# st. lb<=x<=ub, the first num_int variables of x are integer
#
# where obj is the objective function, x0 the initial guess,
# lb and ub are the bounds, num_int is the number of integer variables,
# and max_evals is the maximum number of objective evaluations (rand_evals of these
# are random evaluations).
#
# Laurens Bliek, 06-03-2019
#
# Source: https://github.com/lbliek/MVRSM
# Article: Black-box Mixed-Variable Optimization using a Surrogate Model that Satisfies Integer Constraints,
# 		   by Laurens Bliek, Arthur Guijt, Sicco Verwer, Mathijs de Weerdt

import math
import random
import time
import numpy as np

from scipy.optimize import minimize, Bounds


def relu(x):
    """
    The Rectified Linear Unit (ReLU) function.
    :param x: the input
    """
    return np.maximum(0, x)


def relu_deriv(x):
    """
    The derivative of the rectified linear unit function,
    defined with `relu_deriv(0) = 0.5`.
    :param x: the input
    """
    return (x > 0) + 0.5 * (x == 0)


class SurrogateModel:
    def __init__(self, m, c, W, b, reg, bounds: Bounds, n_obj: int):
        """
        Container for the surrogate model data, defined as a linear combination of
        `m` basis functions whose weights `c` are to be trained. The basis function
        `Φ_k(x)` is a ReLU with input `z_k(x)`, a linear function with weights `W_{k, ·}ᵀ`
        and bias `b_k`.
        Let `d` be the number of (discrete and continuous) decision variables.
        :param m: the number of basis functions.
        :param c: the basis functions weights (m×1 vector).
        :param W: the `z_k(x)` functions weights (m×d matrix).
        :param b: the `z_k(x)` functions biases (m×1 vector).
        :param reg: the regularization parameter.
        :param bounds: the decision variable bounds.
        :param n_obj: number of objectives.
        """
        self.m = m
        self.c = c
        self.W = W
        self.b = b
        self.P = np.diag(np.full(m, 1 / reg))  # RLS covariance matrix
        self.bounds = bounds
        self.n_obj = n_obj

    @classmethod
    def init(cls, d, lb, ub, num_int, n_obj: int) -> 'SurrogateModel':
        """
        Initializes a surrogate model.
        :param d: the number of (discrete and continuous) decision variables.
        :param lb: the lower bound of the decision variable values.
        :param ub: the upper bound of the decision variable values.
        :param num_int: the number of discrete decision variables (`0 ≤ num_int ≤ d`).
        """
        # Define the basis functions parameters.
        W = []  # weights
        b = []  # biases

        # Add a constant basis functions independent of x, giving the model an offset.
        W.append([0] * d)
        b.append(1)

        # Add basis functions dependent on one integer variable
        for k in range(num_int):
            for i in range(int(lb[k]), int(ub[k]) + 1):
                weights = np.zeros(d)
                if i == lb[k]:
                    weights[k] = 1
                    W.append(weights)
                    b.append(-i)
                elif i == ub[k]:
                    weights[k] = -1
                    W.append(weights)
                    b.append(i)
                else:
                    weights[k] = -1
                    W.append(weights)
                    b.append(i)

                    weights = np.zeros(d)
                    weights[k] = 1
                    W.append(weights)
                    b.append(-i)

        # Add basis functions dependent on two subsequent integer variables
        for k in range(1, num_int):
            for i in range(int(lb[k]) - int(ub[k - 1]), int(ub[k]) - int(lb[k - 1]) + 1):
                weights = np.zeros(d)
                if i == lb[k] - ub[k - 1]:
                    weights[k] = 1
                    weights[k - 1] = -1
                    W.append(weights)
                    b.append(-i)
                elif i == ub[k] - lb[k - 1]:
                    weights[k] = -1
                    weights[k - 1] = 1
                    W.append(weights)
                    b.append(i)
                else:
                    weights[k] = -1
                    weights[k - 1] = 1
                    W.append(weights)
                    b.append(i)

                    weights = np.zeros(d)
                    weights[k] = 1
                    weights[k - 1] = -1
                    W.append(weights)
                    b.append(-i)

        # The number of basis functions only related to discrete variables.
        int_basis_count = len(b) - 1

        # Add `num_cont` random linearly independent basis functions (and parallel ones)
        # that depend on both integer and continuous variables, where `num_cont` is
        # the number of continuous variables.
        num_cont = d - num_int
        W_cont = np.random.random((num_cont, d))
        W_cont = (2 * W_cont - 1) / d  # normalize between -1/d and 1/d.
        for k in range(num_cont):
            # Find the set in which `b` needs to lie by moving orthogonal to W.
            signs = np.sign(W_cont[k])

            # Find relevant corner points of the [lb, ub] hypercube.
            corner_1 = np.copy(lb)
            corner_2 = np.copy(ub)
            for j in range(d):
                if signs[j] < 0:
                    corner_1[j] = ub[j]
                    corner_2[j] = lb[j]

            # Calculate minimal distance from hyperplane to corner points.
            b1 = np.dot(W_cont[k], corner_1)
            b2 = np.dot(W_cont[k], corner_2)

            if b1 > b2:
                print('Warning: b1>b2. This may lead to problems.')

            # Add the same number of basis functions as for the discrete variables.
            if num_int>0:

                for j in range(math.ceil(int_basis_count / num_int)):
                    # or just add 1000 of them
                    # for j in range(1000):
                    b_j = (b2 - b1) * np.random.random() + b1
                    W.append(W_cont[k])
                    b.append(-float(b_j))
            else:
                for j in range(500):
                    b_j = (b2 - b1) * np.random.random() + b1
                    W.append(W_cont[k])
                    b.append(-float(b_j))

        W = np.asarray(W)
        b = np.asarray(b)
        m = len(b)  # the number of basis functions

        c = np.zeros((n_obj, m))  # the model weights
        # Set model weights corresponding to discrete basis functions to 1, stimulates convexity.
        c[:,1:int_basis_count + 1] = 1

        # The regularization parameter. 1e-8 is good for the noiseless case,
        # replace by ≈1e-3 if there is noise.
        reg = 1e-4
        return cls(m, c, W, b, reg, Bounds(lb, ub), n_obj)

    def phi(self, x):
        """
        Evaluates the basis functions at `x`.
        :param x: the decision variable values
        """
        z = np.matmul(self.W, x) + self.b
        return relu(z)

    def phi_deriv(self, x):
        """
        Evaluates the derivatives of the basis functions with respect to `x`.
        :param x: the decision variable values
        """
        z = np.matmul(self.W, x) + self.b
        return relu_deriv(z)

    def update(self, x, y):
        """
        Updates the model upon the observation of a new data point `(x, y)`.
        :param x: the decision variables values
        :param y: the objective function value `y(x)`
        """
        phi = self.phi(x)  # basis function values for k = 1, ..., m.

        # Recursive least squares algorithm
        v = np.matmul(self.P, phi)
        g = v / (1 + np.inner(phi, v))
        self.P -= np.outer(g, v)

        for i in range(self.n_obj):
            self.c[i, :] = self.c[i, :] + (y[i] - np.inner(phi, self.c[i, :])) * g

    def g(self, x):
        """
        Evaluates the surrogate model at `x`.
        :param x: the decision variable values.
        """
        phi = self.phi(x)
        #print(self.c @ phi)
        return self.c @ phi

    def g_jac(self, x):
        """
        Evaluates the Jacobian of the model at `x`.
        :param x: the decision variable values.
        """
        phi_prime = self.phi_deriv(x)
        # Treat phi_prime as a column vector to scale the rows w_1, ..., w_m
        # of W by Φ'_1, ..., Φ'_m, respectively.
        W_scaled = phi_prime[:, None] * self.W
        return np.matmul(self.c, W_scaled)

    def g_scalarize(self, x, scalarization_weights):
        """
        Evaluates the linear scalarization of multiple objectives.
        :param x: the decision variable values
        :param scalarization_weights: vector of size n_obj
        """
        # single_obj = inner product between scalarization_weights and vector of g
        # or: single_obj = sum of scalarization_weights[obj_index]*g[obj_index]
        # return single_obj
        return self.g(x) @ scalarization_weights

    def g_scalarize_max(self, x, scalarization_weights):
        """
        Evaluates the maximum or Tchebycheff scalarization of multiple objectives.
        :param x: the decision variable values
        :param scalarization_weights: vector of size n_obj
        """
        # worst objective after multiplying with weight
        mul = np.multiply(self.g(x), scalarization_weights)
        j = np.argmax(mul)
        return self.g(x)[j]


    def g_scalarize_jac(self, x, scalarization_weights):
        """
        Evaluates the Jacobian of the linear scalarization of multiple objectives.
        :param x: the decision variable values
        :param scalarization_weights: vector of size n_obj
        """
        return np.matmul(scalarization_weights,self.g_jac(x))

    def g_scalarize_max_jac(self, x, scalarization_weights):
        """
        Evaluates the Jacobian of the maximum or Tchebycheff scalarization of multiple objectives.
        :param x: the decision variable values
        :param scalarization_weights: vector of size n_obj
        """
        mul = np.multiply(self.g(x), scalarization_weights)
        j = np.argmax(mul)
        return self.g_jac(x)[j,:]

    def augmented_Tchebycheff(self, x, scalarization_weights):
        """
        Evaluates the augmented Tchebycheff scalarization of multiple objectives.
        :param x: the decision variable values
        :param scalarization_weights: vector of size n_obj
        """
        #Augmented Tchebycheff scalarization from ``ParEGO: A Hybrid Algorithm With On-Line Landscape Approximation for Expensive Multiobjective Optimization Problems''
        return self.g_scalarize_max(x, scalarization_weights) + 0.05*self.g_scalarize(x, scalarization_weights)

    def augmented_Tchebycheff_jac(self, x, scalarization_weights):
        """
        Evaluates the Jacobian of the augmented Tchebycheff scalarization of multiple objectives.
        :param x: the decision variable values
        :param scalarization_weights: vector of size n_obj
        """
        return self.g_scalarize_max_jac(x, scalarization_weights) + 0.05*self.g_scalarize_jac(x, scalarization_weights)



    def minimum(self, x0, scalarization_weights):
        """
        Find a minimum of the surrogate model approximately.
        :param x0: the initial guess.
        :param scalarization_weights: weights for the scalarization of multiple objectives
        :return minimization evaluation and corresponding function values
        """
        scalarization_type = 2 #0=linear, 1=max, 0.5 is a mix, 2 is augmented Tchebycheff. max seems to capture the shape of nonconvex pareto front better (also according to theory) but has worse performance
        # Current default is the augmented Tchebycheff, as it is used in other papers.

        if scalarization_type==0:
            # with linear scalarization
            print('Using linear scalarization')
            res = minimize(self.g_scalarize, x0, args=(scalarization_weights,), method='L-BFGS-B', bounds=self.bounds,
                           jac=self.g_scalarize_jac,
                            options={'maxiter': 20, 'maxfun': 20})
        elif scalarization_type==1:
            #with max scalarization
            #Similar to:
            #Golovin, Daniel and Qiuyi Zhang. “Random Hypervolume Scalarizations for Provable Multi-Objective Black Box Optimization.” ArXiv abs/2006.04655 (2020): n. pag.
            print('Using max scalarization')
            res = minimize(self.g_scalarize_max, x0, args=(scalarization_weights,), method='L-BFGS-B', bounds=self.bounds,
                           jac=self.g_scalarize_max_jac,
                           options={'maxiter': 20, 'maxfun': 20})
        elif scalarization_type==0.5:
            #with mix scalarization
            r = random.random()
            if r>0.5:
                print('Using mixed scalarization (now max)')
                res = minimize(self.g_scalarize_max, x0, args=(scalarization_weights,), method='L-BFGS-B', bounds=self.bounds,
                           jac=self.g_scalarize_max_jac,
                           options={'maxiter': 20, 'maxfun': 20})
            elif r<=0.5:
                print('Using mixed scalarization (now linear)')
                res = minimize(self.g_scalarize, x0, args=(scalarization_weights,), method='L-BFGS-B',
                               bounds=self.bounds,
                               jac=self.g_scalarize_jac,
                               options={'maxiter': 20, 'maxfun': 20})
            else:
                print('Warning: wrong random number generated')
        elif scalarization_type==2:
            #with augmented Tchebycheff scalarization (from ``ParEGO: A Hybrid Algorithm With On-Line Landscape Approximation for Expensive Multiobjective Optimization Problems'')
            # Warning: type 2 Augmented Tchebycheff is untested!!!
            print('Using augmented Tchebycheff scalarization')
            res = minimize(self.augmented_Tchebycheff, x0, args=(scalarization_weights,), method='L-BFGS-B', bounds=self.bounds,
                           jac=self.augmented_Tchebycheff_jac,
                           options={'maxiter': 20, 'maxfun': 20})
        else:
            print('Warning: wrong scalarization chosen.')
        return res.x, res.fun





def scale(y, y0, SCALE_THRESHOLD = 1e-8):
    """
    Scale the objective with respect to the initial objective value,
    causing the optimum to lie below zero. This helps exploration and
    prevents the algorithm from getting stuck at the boundary.
    :param y: the objective function value.
    :param y0: the initial objective function value, `y(x0)`.
    """

    y = np.asarray(y)
    y0 = np.asarray(y0)
    y0 = abs(y0)
    y -= y0
    for i in range(len(y)):
        if y0[i] > SCALE_THRESHOLD:
            y[i] /= y0[i]

    return y


def inv_scale(y_scaled, y0, SCALE_THRESHOLD = 1e-8):
    """
    Computes the inverse function of `scale(y, y0)`.
    :param y_scaled: the scaled objective function value.
    :param y0: the initial objective function value, `y(x0)`.
    :return: the value `y` such that `scale(y, y0) = y_scaled`.
    """
    for i in range(len(y0)):
        if abs(y0[i]) > SCALE_THRESHOLD:
            y_scaled[i] *= abs(y0[i])
    return y_scaled + y0







def MVRSM_minimize(obj, x0, lb, ub, num_int: int, max_evals: int, rand_evals: int=0, n_objectives: int=1):
    start_time = time.time()
    log_filename = f'log_MVRSM_{str(start_time)}.log'
    d = len(x0)  # number of decision variables

    model = SurrogateModel.init(d, lb, ub, num_int, n_objectives)
    next_x = x0  # candidate solution



    best_y = math.inf  # least objective function value found so far, equal to obj(best_x).
    best_x = np.copy(next_x)  # best candidate solution found so far
    if n_objectives >=2:
        best_y = [math.inf]*n_objectives  # least objective function value found so far, equal to obj(best_x).

    ylist = [] # All evaluations
    xlist = [] # All evaluated solutions

    # ylist_Pareto = [] # Needed to calculate Pareto front
    # xlist_Pareto = [] # All Pareto optimal solutions

    if n_objectives >= 2:
        Pareto_index = [] # List of the iterations of which evaluated solutions are currently Pareto optimal


    # Iteratively evaluate the objective, update the model, find the minimum of the model,
    # and explore the search space.
    for i in range(0, max_evals):
        iter_start = time.time()
        print(f'Starting MVRSM iteration {i}/{max_evals}')


        # Evaluate the objective and scale it.
        x = np.copy(next_x).astype(float)
        y_unscaled = obj(x)
        ylist.append(y_unscaled)
        xlist.append(x)
        if i == 0:
            y0 = y_unscaled
        y = scale(y_unscaled, y0)


        # Keep track of Pareto front
        if n_objectives >= 2:
            if i==0:
                Pareto_index.append(i)
            else:
                y_is_dominated = 0 # whether y is dominated by any of the previous Pareto optimal solutions
                pp_to_remove = [] # Pareto optimal points that should be removed because they are not Pareto optimal anymore due to y
                for pp in Pareto_index:
                    y_dominates_pp = 0
                    pp_dominates_y = 0
                    TOL = 1e-8 #tolerance level for determining Pareto optimality
                    y_pp_scaled = scale(ylist[pp],y0)

                    # Check if y is exactly the same as pp, in which case it is not useful to add it to the list of Pareto optimal solutions
                    if (y_pp_scaled == y).all():
                        y_is_dominated = 1
                        break
                    for obj_i in range(n_objectives):
                        #Situation A: y is dominated by pp
                        diff = y_pp_scaled[obj_i] - y[obj_i]
                        if diff < -TOL:
                            pp_dominates_y += 1


                        #Situation B: pp dominated by y
                        if diff > TOL:
                            y_dominates_pp += 1
                    # print('pp_dominates_y', pp_dominates_y)
                    # print('y_dominates_pp', y_dominates_pp)
                    #Situation A: y is dominated by pp
                    if pp_dominates_y >= 1 and y_dominates_pp == 0:
                        #keep list the same, and don't add y
                        #y is dominated by pp so stop checking other pp
                        y_is_dominated = 1
                        #print('A')
                        break

                    #Situation B: pp dominated by y
                    if y_dominates_pp >= 1 and pp_dominates_y == 0:
                        pp_to_remove.append(pp)
                        #print('B')
                for pp in pp_to_remove:
                    Pareto_index.remove(pp)
                #Situation C: pp and y don't dominate each other, for every pp --> y is then Pareto optimal
                if y_is_dominated == 0:
                    Pareto_index.append(i)

        #print(Pareto_index)

        # Update the surrogate model
        update_start = time.time()
        model.update(x, y)
        update_time = time.time() - update_start

        # Get scalarization weights
        rnd_weights = np.random.rand(n_objectives)
        scalarization_weights = rnd_weights / rnd_weights.sum()


        # Pick a random Pareto optimal x to start optimization from
        if n_objectives >= 2:
            rand_pp = np.random.randint(len(Pareto_index))
            best_x = xlist[Pareto_index[rand_pp]]
        if n_objectives == 1:
            if y < best_y:
                best_x = np.copy(x)
                best_y = y

        # Minimize surrogate model
        min_start = time.time()
        next_x, min_y = model.minimum(best_x, scalarization_weights) #start from best_x, from a random Pareto optimal solution
        minimization_time = time.time() - min_start



        # Round discrete variables to the nearest integer.
        next_x_before_rounding = np.copy(next_x)
        next_x[0:num_int].round(out=next_x[0:num_int])

        # Visualize model

        # if ii > max_evals/2:

        # import matplotlib.pyplot as plt
        # # print('Hoi', len(xxxx))
        # # print(len(toplot))
        # # plt.plot(xxxx,toplot)
        # # #plt.plot(jjjj,toplot[iiii],'*')
        # # titlestr = ['Dimension ', iiii]
        # # plt.title(titlestr)
        # # plt.show()

        # from mpl_toolkits.mplot3d import Axes3D
        # from matplotlib import cm
        # from matplotlib.ticker import LinearLocator, FormatStrFormatter
        # #XX = np.arange(next_X[0]-0.5, next_X[0]+0.5, 0.01)
        # #XX = np.arange(lb[0], ub[0], 0.05)
        # #YY = np.arange(lb[1], ub[1], 0.05)
        # XX = np.arange(-2, 4, 0.05)
        # YY = np.arange(-2, 4, 0.05)
        # XXX, YYY = np.meshgrid(XX, YY)
        # R = []
        # for XXXX in XX:
        # temp = []
        # for YYYY in YY:
        # #print(next_X)
        # temp.append(model.g([XXXX,YYYY]))
        # R.append(temp)
        # R = np.copy(R)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(XXX, YYY, R, cmap=cm.coolwarm,
        # linewidth=0, antialiased=False)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()

        # Just to be sure, clip the decision variables to the bounds.
        np.clip(next_x, lb, ub, out=next_x)

        # Check if minimizer really gives better result
        # if model.g(next_X) > model.g(x) + 1e-8:
        # print('Warning: minimization of the surrogate model in MVRSM yielded a worse solution, maybe something went wrong.')

        # Perform exploration to prevent the algorithm from getting stuck in local minima
        # of the surrogate model.
        next_x_before_exploration = np.copy(next_x)
        next_x = np.copy(next_x)
        if i < rand_evals:
            # Perform random search
            next_x[0:num_int] = np.random.randint(lb[0:num_int], np.add(ub[0:num_int], [1] * num_int))  # high is exclusive
            next_x[num_int:d] = np.random.uniform(lb[num_int:d], ub[num_int:d])
        # Skip exploration in the last n_objectives iterations (to end at the Pareto Front of the surrogate model).
        elif i < max_evals - n_objectives:
            # Randomly perturb the discrete variables. Each x_i is shifted n units
            # to the left (if dir is False) or to the right (if dir is True).
            # The bounds of each variable are respected.
            int_pert_prob = 1 / d  # probability that x_i is permuted
            #int_pert_prob = 0.3  # probability that x_i is permuted
            for j in range(num_int):
                r = random.random()  # determines n
                direction = random.getrandbits(1)  # whether to explore towards -∞ or +∞
                value = next_x[j]
                while r < int_pert_prob:
                    if lb[j] == value < ub[j]:
                        value += 1
                    elif lb[j] < value == ub[j]:
                        value -= 1
                    elif lb[j] < value < ub[j]:
                        value += 1 if direction else -1
                    r *= 2
                next_x[j] = value

            # Continuous exploration
            for j in range(num_int, d):
                value = next_x[j]
                while True:  # re-sample while out of bounds.
                    # Choose a variance that scales inversely with the number of decision variables.
                    # Note that Var(aX) = a^2 Var(X) for any random variable.
                    delta = np.random.normal() * (ub[j] - lb[j]) * 0.1 / math.sqrt(d)
                    if lb[j] <= value + delta <= ub[j]:
                        next_x[j] += delta
                        break

            # Just to be sure, clip the decision variables to the bounds again.
            np.clip(next_x, lb, ub, out=next_x)

        iter_time = time.time() - iter_start

        # Save data to log file
        with open(log_filename, 'a') as f:
            print('\n\n MVRSM iteration: ', i, file=f)
            print('Time spent training the model:				 ', update_time, file=f)
            print('Time spent finding the minimum of the model: ', minimization_time, file=f)
            print('Total computation time for this iteration:	', iter_time, file=f)
            print('Current time: ', time.time(), file=f)
            print('Evaluated data point and evaluation:						   ', np.copy(x).astype(float), ', ',
                  inv_scale(y, y0), file=f)
            print('Predicted value at evaluated data point (after learning)       ', np.copy(x).astype(float), ', ',
                  inv_scale(model.g(x), y0), file=f)
            print('Best found data point and evaluation so far:				   ', np.copy(best_x).astype(float),
                  ', ', inv_scale(best_y, y0), file=f)
            print('Best data point according to the model and predicted value:	   ', next_x_before_rounding, ', ',
                  inv_scale(model.g(next_x_before_rounding), y0), file=f)
            print('Best rounded	 point according to the model and predicted value:', next_x_before_exploration, ', ',
                  inv_scale(model.g(next_x_before_exploration), y0), file=f)
            print('Suggested next data point and predicted value:				   ', next_x, ', ',
                  inv_scale(model.g(next_x), y0), file=f)
            if i >= max_evals - 1:
                np.set_printoptions(threshold=np.inf)
                print('Model c parameters: ', np.transpose(model.c), file=f)
                print('Model W parameters: ', np.transpose(model.W), file=f)
                print('Model B parameters: ', np.transpose(model.b), file=f)
                np.set_printoptions(threshold=1000)
    if n_objectives == 1:
        with open(log_filename, 'a') as f:
            print("\n\nList of evaluations:", file=f)
            print(f"X = {xlist}", file=f)
            print(f"Y = {ylist}", file=f)
            print("Solution found: ", file=f)
            print(f"X = {best_x}", file=f)
            print(f"Y = {inv_scale(best_y, y0)}", file=f)
        return xlist, ylist, best_x, inv_scale(best_y, y0), model, log_filename
    if n_objectives >= 2:
        # return all solutions, and Pareto optimal solutions
        with open(log_filename, 'a') as f:
            print("\n\nList of evaluations:", file=f)
            print(f"X = {xlist}", file=f)
            print(f"Y = {ylist}", file=f)
            print("Pareto front found: ", file=f)
            print(f"X = {[xlist[iii] for iii in Pareto_index]}", file=f)
            print(f"Y = {[ylist[iii] for iii in Pareto_index]}", file=f)
        return xlist, ylist, [xlist[iii] for iii in Pareto_index], [ylist[iii] for iii in Pareto_index], model, log_filename


def read_log(filename):
    """
    Read data from log file (this reads the best found objective values
    at each iteration).
    :param filename: the log filename.
    """
    with open(filename, 'r') as f:
        mvrsm_file = f.readlines()
        best_y = []
        for i, lines in enumerate(mvrsm_file):
            search_term = 'Best data point according to the model and predicted value'
            if search_term in lines:
                # print('Hello', MVRSMfile)
                temp = mvrsm_file[i - 1]
                temp = temp.split('] , ')
                temp = temp[1]
                best_y.append(float(temp))
    return best_y


def plot_results(filename):
    """
    Plot the best found objective values at each iteration.
    :param filename: the log filename.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    best_y = read_log(filename)
    plt.plot(best_y)
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.grid()
    plt.show()


# fig.show()

def visualise_model(model, obj, x0, lb, ub, num_int):
    ## Plot in 'one dimension' (first integer and first continuous variable)
    # print(num_int)
    # print(ub[num_int]-lb[num_int])
    # print((ub[num_int]-lb[num_int])/0.05)
    print('W parameters: ', model.W)
    print('B parameters: ', model.b)
    int_range = np.arange(lb[0], ub[0], 0.05)  # range of the integer variable
    cont_range = np.arange(lb[num_int], ub[num_int], 0.05)  # range of the continuous variable
    model_output = np.zeros((len(int_range), len(cont_range)))
    obj_output = np.zeros((len(int_range), len(cont_range)))
    x = np.copy(x0)  # For the other variables, use x0 as the value
    correctint = 999
    for i in range(len(int_range)):
        x[0] = int_range[i]
        if abs(x[0] + 10) <= 0.05:
            correctint = i
        for j in range(len(cont_range)):
            x[num_int] = cont_range[j]
            model_output[i, j] = model.inv_scale(model.g(x))
            obj_output[i, j] = obj(x)
    X, Y = np.meshgrid(int_range, cont_range)
    R = np.sqrt(X ** 2 + Y ** 2)
    R = np.sin(R)
    # print(R.shape)
    # print(model_output.shape)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure(2)
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, np.transpose(model_output), cmap=cm.coolwarm)
    ax.set_title('Model output')
    ax.set_xlabel('Discrete variable')
    ax.set_ylabel('Continuous variable')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, np.transpose(obj_output), cmap=cm.coolwarm)
    ax2.set_title('Function output')
    ax2.set_xlabel('Discrete variable')
    ax2.set_ylabel('Continuous variable')
    fig.show()

    fig2 = plt.figure(3)
    ax3 = fig2.add_subplot(121)
    ax3.plot(cont_range, np.transpose(model_output[correctint, :]))
    ax3.set_title('Model output')
    # ax3.set_xlabel('Discrete variable')
    ax3.set_xlabel('Continuous variable')
    ax4 = fig2.add_subplot(122)
    ax4.plot(cont_range, np.transpose(obj_output[correctint, :]))
    ax4.set_title('Function output')
    ax4.set_xlabel('Continuous variable')
    fig2.show()

    plt.show()

# fig2 = plt.figure()
# ax3 = fig2.add_subplot(121)
# cs3 = ax3.contourf(X, Y, np.transpose(model_output),cmap=cm.coolwarm)
# ax3.contour(cs3)
# ax3.set_title('Model output')
# ax3.set_xlabel('Discrete variable')
# ax3.set_ylabel('Continuous variable')
# ax4 = fig2.add_subplot(122)
# cs4 = ax4.contourf(X, Y, np.transpose(obj_output),cmap=cm.coolwarm)
# ax4.contour(cs4)
# ax4.set_title('Function output')
# ax4.set_xlabel('Discrete variable')
# ax4.set_ylabel('Continuous variable')
# plt.show()
