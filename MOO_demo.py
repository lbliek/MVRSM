# MVRSM demo
# By Laurens Bliek, 16-03-2020
# Supported functions: 'func2C', 'func3C', 'dim10Rosenbrock',
# 'linearmivabo', 'dim53Rosenbrock', 'dim53Ackley', 'dim238Rosenbrock'
# Example: python demo.py -f dim10Rosenbrock  -n 10 -tl 4
# Here, -f is the function to be optimised, -n is the number of iterations, and -tl is the total number of runs.
# Afterward, use plot_result.py for visualisation.

import sys
# sys.path.append('../bayesopt')
# sys.path.append('../ml_utils')
import argparse
import os
import numpy as np
import pickle
import time
import testFunctions.syntheticFunctions
# from methods.CoCaBO import CoCaBO
# from methods.BatchCoCaBO import BatchCoCaBO
import MVRSM
import MOO_MVRSM
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from functools import partial
import matplotlib.pyplot as plt
import matplotlib


from scipy.optimize import rosen
from linear_MIVABOfunction import Linear



matplotlib.use('TkAgg')

# CoCaBO code taken from:
# -*- coding: utf-8 -*-
#==========================================
# Title:  run_cocabo_exps.py
# Author: Binxin Ru and Ahsan Alvi
# Date:	  20 August 2019
# Link:	  https://arxiv.org/abs/1906.08878
#==========================================

# =============================================================================
#  CoCaBO Algorithms 
# =============================================================================
#
# def CoCaBO_Exps(obj_func, budget, initN=24 ,trials=40, kernel_mix = 0.5, batch=None):
#
# 	# define saving path for saving the results
# 	saving_path = f'data/syntheticFns/{obj_func}/'
# 	if not os.path.exists(saving_path):
# 		os.makedirs(saving_path)
#
# 	# define the objective function
# 	if obj_func == 'func2C':
# 		f = testFunctions.syntheticFunctions.func2C
# 		categories = [3, 5]
#
# 		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2)},
# 			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
# 			{'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]
#
# 	elif obj_func == 'func3C':
# 		f = testFunctions.syntheticFunctions.func3C
# 		categories = [3, 5, 4]
#
# 		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2)},
# 			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
# 			{'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
# 			{'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]
# 	#Adapted
# 	elif obj_func == 'highdimRosenbrock':
# 		f = testFunctions.syntheticFunctions.highdimRosenbrock
# 		categories = [5,5,5,5,5]
# 		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'h4', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'h5', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'x1', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x2', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x3', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x4', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x5', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x6', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x7', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x8', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x9', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x10', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x11', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x12', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x13', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x14', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x15', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x16', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x17', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x18', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x19', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x20', 'type': 'continuous', 'domain': (-2, 2)}]
# 	elif obj_func == 'dim10Rosenbrock':
# 		f = testFunctions.syntheticFunctions.dim10Rosenbrock
# 		categories = [5,5,5]
# 		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
# 			{'name': 'x1', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x2', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x3', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x4', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x5', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x6', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x7', 'type': 'continuous', 'domain': (-2, 2)}]
# 	elif obj_func == 'dim53Rosenbrock':
# 		f = testFunctions.syntheticFunctions.dim53Rosenbrock
# 		categories = []
# 		for i in range(50):
# 			categories.append(2)
# 		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h3', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h4', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h5', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h6', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h7', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h8', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h9', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h10', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h11', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h12', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h13', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h14', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h15', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h16', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h17', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h18', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h19', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h20', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h21', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h22', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h23', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h24', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h25', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h26', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h27', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h28', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h29', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h30', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h31', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h32', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h33', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h34', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h35', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h36', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h37', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h38', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h39', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h40', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h41', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h42', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h43', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h44', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h45', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h46', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h47', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h48', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h49', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'h50', 'type': 'categorical', 'domain': (0, 1)},
# 			{'name': 'x1', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x2', 'type': 'continuous', 'domain': (-2, 2)},
# 			{'name': 'x3', 'type': 'continuous', 'domain': (-2, 2)}]
# 	elif obj_func == 'dim238Rosenbrock':
# 		f = testFunctions.syntheticFunctions.dim238Rosenbrock
# 		categories = []
# 		bounds = []
# 		for i in range(119):
# 			categories.append(5)
# 			bounds.append({'name': f"h{i}", 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)})
# 		for i in range(119,238):
# 			bounds.append({'name': f"x{i-119+1}", 'type': 'continuous', 'domain': (-2,2)})
# 	elif obj_func == 'dim53Ackley':
# 		f = testFunctions.syntheticFunctions.dim53Ackley
# 		categories = []
# 		bounds = []
# 		for i in range(50):
# 			categories.append(2)
# 			bounds.append({'name': f"h{i}", 'type': 'categorical', 'domain': (0, 1)})
# 		for i in range(50,53):
# 			bounds.append({'name': f"x{i-50+1}", 'type': 'continuous', 'domain': (-1, 1)})
# 	elif obj_func == 'linearmivabo':
#
# 		ftemp = LM.objective_function
# 		def f(ht_list, X):
# 			XX = []
# 			for i in ht_list:
# 				XX.append(i)
# 			for i in X:
# 				XX.append(i)
# 			return ftemp(XX)
# 		categories = [3, 3, 3, 3, 3, 3, 3, 3]
# 		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
# 			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
# 			{'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
# 			{'name': 'h4', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
# 			{'name': 'h5', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
# 			{'name': 'h6', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
# 			{'name': 'h7', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
# 			{'name': 'h8', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
# 			{'name': 'x1', 'type': 'continuous', 'domain': (0, 3)},
# 			{'name': 'x2', 'type': 'continuous', 'domain': (0, 3)},
# 			{'name': 'x3', 'type': 'continuous', 'domain': (0, 3)},
# 			{'name': 'x4', 'type': 'continuous', 'domain': (0, 3)},
# 			{'name': 'x5', 'type': 'continuous', 'domain': (0, 3)},
# 			{'name': 'x6', 'type': 'continuous', 'domain': (0, 3)},
# 			{'name': 'x7', 'type': 'continuous', 'domain': (0, 3)},
# 			{'name': 'x8', 'type': 'continuous', 'domain': (0, 3)},]
# 	#/Adapted
# 	else:
# 		raise NotImplementedError
#
# 	# Run CoCaBO Algorithm
# 	if batch == 1:
# 		# sequential CoCaBO
# 		mabbo = CoCaBO(objfn=f, initN=initN, bounds=bounds,
# 					   acq_type='LCB', C=categories,
# 					   kernel_mix = kernel_mix)
#
# 	else:
# 		# batch CoCaBO
# 		mabbo = BatchCoCaBO(objfn=f, initN=initN, bounds=bounds,
# 							acq_type='LCB', C=categories,
# 							kernel_mix=kernel_mix,
# 							batch_size=batch)
# 	mabbo.runTrials(trials, budget, saving_path)









if __name__ == '__main__':

	# Read arguments
	
	parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
	parser.add_argument('-f', '--func', help='Objective function',
						default='MO_Kursawe', type=str)   # Supported functions: 'func2C', 'func3C', 'dim10Rosenbrock',
														       # 'linearmivabo', 'dim53Rosenbrock', 'dim53Ackley', 'dim238Rosenbrock'
	parser.add_argument('-mix', '--kernel_mix',
						help='Mixture weight for production and summation kernel. Default = 0.0', default=0.5,
						type=float)
	parser.add_argument('-n', '--max_itr', help='Max Optimisation iterations. Default = 100',
						default=10, type=int)
	parser.add_argument('-tl', '--trials', help='Number of random trials. Default = 20',
						default=1, type=int)
	parser.add_argument('-b', '--batch', help='Batch size (>1 for batch CoCaBO and =1 for sequential CoCaBO). Default = 1',
						default=1, type=int)

	args = parser.parse_args()
	print(f"Got arguments: \n{args}")
	obj_func = args.func
	kernel_mix = args.kernel_mix
	n_itrs = args.max_itr
	n_trials = args.trials
	batch = args.batch
	
	

	
	
	folder = os.path.join(os.path.curdir, 'data',  'syntheticFns', obj_func)
	if not os.path.isdir(folder):
		os.makedirs(folder)

	num_objectives = 1
	if obj_func == 'dim10Rosenbrock':
		ff = testFunctions.syntheticFunctions.dim10Rosenbrock
		d = 10 # Total number of variables
		lb = -2*np.ones(d).astype(int) # Lower bound
		ub = 2*np.ones(d).astype(int) # Upper bound
		num_int = 3 # number of integer variables
		lb[0:num_int] = 0
		ub[0:num_int] = num_int+1
	elif obj_func == 'func3C':
		ff = testFunctions.syntheticFunctions.func3C
		d = 5 # Total number of variables			
		lb = -1*np.ones(d).astype(int) # Lower bound for continuous variables
		ub = 1*np.ones(d).astype(int) # Upper bound for continuous variables
		num_int = 3 # number of integer variables
		lb[0:num_int] = 0
		ub[0]=2
		ub[1]=4
		ub[2]=3
	elif obj_func == 'func2C':
		ff = testFunctions.syntheticFunctions.func2C
		d = 4 # Total number of variables			
		lb = -1*np.ones(d).astype(int) # Lower bound for continuous variables
		ub = 1*np.ones(d).astype(int) # Upper bound for continuous variables
		num_int = 2 # number of integer variables
		lb[0:num_int] = 0
		ub[0]=2
		ub[1]=4
	elif obj_func == 'linearmivabo':
		LM = Linear(laplace=False)
		ff = LM.objective_function
		d = 16 # Total number of variables			
		lb = 0*np.ones(d).astype(int) # Lower bound for continuous variables
		ub = 3*np.ones(d).astype(int) # Upper bound for continuous variables
		num_int = 8 # number of integer variables
		lb[0:num_int] = 0
		ub[0:num_int]=3
	elif obj_func == 'dim53Rosenbrock':
		ff = testFunctions.syntheticFunctions.dim53Rosenbrock
		d = 53 # Total number of variables
		lb = -2*np.ones(d).astype(int) # Lower bound
		ub = 2*np.ones(d).astype(int) # Upper bound
		num_int = 50 # number of integer variables
		lb[0:num_int] = 0
		ub[0:num_int] = 1
	elif obj_func == 'dim53Ackley':
		ff = testFunctions.syntheticFunctions.dim53Ackley
		d = 53 # Total number of variables
		lb = -1*np.ones(d).astype(float) # Lower bound
		ub = 1*np.ones(d).astype(float) # Upper bound
		num_int = 50 # number of integer variables
		lb[0:num_int] = 0
		ub[0:num_int] = 1
	elif obj_func == 'dim238Rosenbrock':
		ff = testFunctions.syntheticFunctions.dim238Rosenbrock
		d = 238 # Total number of variables
		lb = -2*np.ones(d).astype(int) # Lower bound
		ub = 2*np.ones(d).astype(int) # Upper bound
		num_int = 119 # number of integer variables
		lb[0:num_int] = 0
		ub[0:num_int] = 4
	elif obj_func == 'MO_dim53Rosenbrock_Ackley':
		ff1 = testFunctions.syntheticFunctions.dim53Rosenbrock
		ff2 = testFunctions.syntheticFunctions.dim53Ackley
		num_objectives = 2
		d = 53  # Total number of variables
		lb = -2 * np.ones(d).astype(int)  # Lower bound
		ub = 2 * np.ones(d).astype(int)  # Upper bound
		num_int = 50  # number of integer variables
		lb[0:num_int] = 0
		ub[0:num_int] = 1
	elif obj_func == 'MO_Kursawe':
		ff1 = testFunctions.syntheticFunctions.Kursawe1
		ff2 = testFunctions.syntheticFunctions.Kursawe2
		num_objectives = 2
		d = 3  # Total number of variables
		lb = -5 * np.ones(d).astype(int)  # Lower bound
		ub = 5 * np.ones(d).astype(int)  # Upper bound
		num_int = 0  # number of integer variables
		#lb[0:num_int] = 0
		#ub[0:num_int] = 1
	elif obj_func == 'MO_ZDT3':
		ff1 = testFunctions.syntheticFunctions.ZDT3_1
		ff2 = testFunctions.syntheticFunctions.ZDT3_2
		num_objectives = 2
		d = 30  # Total number of variables
		lb = 0 * np.ones(d).astype(int)  # Lower bound
		ub = 1 * np.ones(d).astype(int)  # Upper bound
		num_int = 0  # number of integer variables


	else:
		raise NotImplementedError
	
	
	x0 =np.zeros(d) # Initial guess
	x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
	x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
	
	
	rand_evals = 24 # Number of random iterations, same as initN above (24)
	max_evals = n_itrs+rand_evals # Maximum number of MVRSM iterations, the first <rand_evals> are random
	
	
	###########
	## MVRSM ##
	###########
	
	def obj_MVRSM(x):
		#print(x[0:num_int])
		if num_objectives == 1:
			h = np.copy(x[0:num_int]).astype(int)
			if obj_func == 'func3C' or obj_func == 'func2C':
				result = ff(h,x[num_int:])[0][0]
			elif obj_func == 'linearmivabo':
				result = ff(x)
			else:
				result = ff(h,x[num_int:])
		elif num_objectives == 2:
			h = np.copy(x[0:num_int]).astype(int)
			if obj_func == 'MO_dim53Rosenbrock_Ackley':
				result = [ff1(h,x[num_int:]), ff2(h,x[num_int:])]
			else:
				result = [ff1(x), ff2(x)]
		return result
	def run_MVRSM():
		if num_objectives ==1:
			solX, solY, model, logfile = MVRSM.MVRSM_minimize(obj_MVRSM, x0, lb, ub, num_int, max_evals, rand_evals)
		else:
			solX, solY, model, logfile = MOO_MVRSM.MVRSM_minimize(obj_MVRSM, x0, lb, ub, num_int, max_evals, rand_evals, num_objectives)
		os.rename(logfile, os.path.join(folder,logfile))
		print("Solution found: ")
		print(f"X = {solX}")
		print(f"Y = {solY}")



	for i in range(n_trials):
		if obj_func == 'dim10Rosenbrock' or obj_func == 'dim53Rosenbrock' or obj_func == 'dim238Rosenbrock':
			print(f"Testing MVRSM on the {d}-dimensional Rosenbrock function with integer constraints.")
			print("The known global minimum is f(1,1,...,1)=0")
		else:
			print("Start MVRSM trials")
		run_MVRSM()
		
		
	##############
	## HyperOpt ##
	##############
	
	
	
	# HyperOpt and RS objective
	# def hyp_obj(x):
	# 	f = obj_MVRSM(x)
	# 	#print('Objective value: ', f)
	# 	return {'loss': f, 'status': STATUS_OK }
	#
	# # Two algorithms used within HyperOpt framework (random search and TPE)
	# algo = rand.suggest
	# algo2 = partial(tpe.suggest, n_startup_jobs=rand_evals)
	#
	# # Define search space for HyperOpt
	# var = [ None ] * d #variable for hyperopt and random search
	# for i in list(range(0,d)):
	# 	if i<num_int:
	# 		var[i] = hp.quniform('var_d'+str(i), lb[i], ub[i], 1) # Integer variables
	# 	else:
	# 		var[i] = hp.uniform('var_c'+str(i), lb[i], ub[i]) # Continuous variables
	#
	#
	#
	# print("Start HyperOpt trials")
	# for i in range(n_trials):
	# 	current_time = time.time() # time when starting the HO and RS algorithm
	#
	#
	# 	trials_HO = Trials()
	# 	time_start = time.time() # Start timer
	# 	hypOpt = fmin(hyp_obj, var, algo2, max_evals=max_evals, trials=trials_HO) # Run HyperOpt
	# 	total_time_HypOpt = time.time()-time_start # End timer
	#
	# 	logfileHO = os.path.join(folder, 'log_HypOpt_'+ str(current_time) + ".log")
	# 	with open(logfileHO, 'a') as f:
	# 		print(trials_HO.trials, file=f) # Save log
	#
	#
	# 	#write times per iteration to log
	# 	logHOtimeperiteration = os.path.join(folder, 'HO_timeperiteration.txt')
	# 	with open(logHOtimeperiteration, 'a') as f:
	# 		for i in range(0,max_evals):
	# 			if i==0:
	# 				#print(trials_HO.trials[i]['book_time'].timestamp()+3600- time_start, file=f) #something wrong with my clock which causes 1 hour difference
	# 				print(trials_HO.trials[i]['book_time'].timestamp()- time_start, file=f) #no 1 hour difference
	# 			else:
	# 				print((trials_HO.trials[i]['book_time']- trials_HO.trials[i-1]['book_time']).total_seconds(), file=f)

	

	###################
	## Random search ##
	###################
	#
	# print("Start Random Search trials")
	# for i in range(n_trials):
	# 	current_time = time.time() # time when starting the HO and RS algorithm
	# 	trials_RS = Trials()
	#
	# 	time_start = time.time()
	# 	RS = fmin(hyp_obj, var, algo, max_evals=max_evals, trials = trials_RS)
	# 	total_time_RS = time.time()-time_start
	#
	# 	logfileRS = os.path.join(folder, 'log_RS_'+ str(current_time) + ".log")
	# 	with open(logfileRS, 'a') as f:
	# 		print(trials_RS.trials, file=f) # Save log
	#
	# 	#write times per iteration to log
	# 	logRStimeperiteration = os.path.join(folder, 'RS_timeperiteration.txt')
	# 	with open(logRStimeperiteration, 'a') as f:
	# 		for i in range(0,max_evals):
	# 			if i==0:
	# 				#print(trials_RS.trials[i]['book_time'].timestamp()+3600- time_start, file=f) #something wrong with my clock which causes 1 hour difference, but not with daylight saving time...
	# 				print(trials_RS.trials[i]['book_time'].timestamp()- time_start, file=f) #no 1 hour difference
	# 			else:
	# 				print((trials_RS.trials[i]['book_time']- trials_RS.trials[i-1]['book_time']).total_seconds(), file=f)
	#
	############
	## CoCaBO ##
	############
	
	# print("Start CoCaBO trials")
	# CoCaBO_Exps(obj_func=obj_func, budget=n_itrs,trials=n_trials, kernel_mix = kernel_mix, batch=batch)



		
