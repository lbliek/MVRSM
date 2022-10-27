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
    def __init__(self, m, c, W, b, reg, bounds: Bounds):
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
        """
        self.m = m
        self.c = c
        self.W = W
        self.b = b
        self.P = np.diag(np.full(m, 1 / reg))  # RLS covariance matrix
        self.bounds = bounds

    @classmethod
    def init(cls, d, lb, ub, num_int) -> 'SurrogateModel':
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
            for j in range(math.ceil(int_basis_count / num_int)):
                # or just add 1000 of them
                # for j in range(1000):
                b_j = (b2 - b1) * np.random.random() + b1
                W.append(W_cont[k])
                b.append(-float(b_j))

        W = np.asarray(W)
        b = np.asarray(b)
        m = len(b)  # the number of basis functions

        c = np.zeros(m)  # the model weights
        # Set model weights corresponding to discrete basis functions to 1, stimulates convexity.
        c[1:int_basis_count + 1] = 1

        # The regularization parameter. 1e-8 is good for the noiseless case,
        # replace by ≈1e-3 if there is noise.
        reg = 1e-8
        return cls(m, c, W, b, reg, Bounds(lb, ub))

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
        self.c += (y - np.inner(phi, self.c)) * g

    def g(self, x):
        """
        Evaluates the surrogate model at `x`.
        :param x: the decision variable values.
        """
        phi = self.phi(x)
        return np.inner(self.c, phi)

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

    def minimum(self, x0):
        """
        Find a minimum of the surrogate model approximately.
        :param x0: the initial guess.
        """
        res = minimize(self.g, x0, method='L-BFGS-B', bounds=self.bounds, jac=self.g_jac,
                       options={'maxiter': 20, 'maxfun': 20})
        return res.x


SCALE_THRESHOLD = 1e-8


def scale(y, y0):
    """
    Scale the objective with respect to the initial objective value,
    causing the optimum to lie below zero. This helps exploration and
    prevents the algorithm from getting stuck at the boundary.
    :param y: the objective function value.
    :param y0: the initial objective function value, `y(x0)`.
    """
    y -= y0
    if abs(y0) > SCALE_THRESHOLD:
        y /= abs(y0)
    return y


def inv_scale(y_scaled, y0):
    """
    Computes the inverse function of `scale(y, y0)`.
    :param y_scaled: the scaled objective function value.
    :param y0: the initial objective function value, `y(x0)`.
    :return: the value `y` such that `scale(y, y0) = y_scaled`.
    """
    if abs(y0) > SCALE_THRESHOLD:
        y_scaled *= abs(y0)
    return y_scaled + y0


def MVRSM_minimize(obj, x0, lb, ub, num_int, max_evals, rand_evals=0):
    start_time = time.time()
    log_filename = f'log_MVRSM_{str(start_time)}.log'
    d = len(x0)  # number of decision variables

    model = SurrogateModel.init(d, lb, ub, num_int)
    next_x = x0  # candidate solution
    best_x = np.copy(next_x)  # best candidate solution found so far
    best_y = math.inf  # least objective function value found so far, equal to obj(best_x).

    # Iteratively evaluate the objective, update the model, find the minimum of the model,
    # and explore the search space.
    for i in range(0, max_evals):
        iter_start = time.time()
        print(f'Starting MVRSM iteration {i}/{max_evals}')

        # Evaluate the objective and scale it.
        x = np.copy(next_x).astype(float)
        y_unscaled = obj(x)
        if i == 0:
            y0 = y_unscaled
        # noinspection PyUnboundLocalVariable
        y = scale(y_unscaled, y0)

        # Keep track of the best found objective value and candidate solution so far.
        if y < best_y:
            best_x = np.copy(x)
            best_y = y

        # Update the surrogate model
        update_start = time.time()
        model.update(x, y)
        update_time = time.time() - update_start

        # Minimize surrogate model
        min_start = time.time()
        next_x = model.minimum(best_x)
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
            next_x[0:num_int] = np.random.randint(lb[0:num_int], ub[0:num_int] + 1)  # high is exclusive
            next_x[num_int:d] = np.random.uniform(lb[num_int:d], ub[num_int:d])
        # Skip exploration in the last iteration (to end at the exact minimum of the surrogate model).
        elif i < max_evals - 2:
            # Randomly perturb the discrete variables. Each x_i is shifted n units
            # to the left (if dir is False) or to the right (if dir is True).
            # The bounds of each variable are respected.
            int_pert_prob = 1 / d  # probability that x_i is permuted
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

    return best_x, inv_scale(best_y, y0), model, log_filename


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
