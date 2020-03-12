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
import sys
# sys.path.append('../bayesopt')
# sys.path.append('../ml_utils')
import argparse
import os
import testFunctions.syntheticFunctions
from methods.CoCaBO import CoCaBO
from methods.BatchCoCaBO import BatchCoCaBO
import MVDONE
import numpy as np
import os
import pickle
from scipy.optimize import rosen


def CoCaBO_Exps(obj_func, budget, initN=24 ,trials=40, kernel_mix = 0.5, batch=None):

	# define saving path for saving the results
	saving_path = f'data/syntheticFns/{obj_func}/'
	if not os.path.exists(saving_path):
		os.makedirs(saving_path)

	# define the objective function
	if obj_func == 'func2C':
		f = testFunctions.syntheticFunctions.func2C
		categories = [3, 5]

		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2)},
			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
			{'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]

	elif obj_func == 'func3C':
		f = testFunctions.syntheticFunctions.func3C
		categories = [3, 5, 4]

		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2)},
			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
			{'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
			{'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]
	#Laurens
	elif obj_func == 'highdimRosenbrock':
		f = testFunctions.syntheticFunctions.highdimRosenbrock
		categories = [5,5,5,5,5]
		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'h4', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'h5', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'x1', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x2', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x3', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x4', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x5', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x6', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x7', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x8', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x9', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x10', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x11', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x12', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x13', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x14', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x15', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x16', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x17', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x18', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x19', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x20', 'type': 'continuous', 'domain': (-2, 2)}]
	#/Laurens
	#Laurens
	elif obj_func == 'dim10Rosenbrock':
		f = testFunctions.syntheticFunctions.dim10Rosenbrock
		categories = [5,5,5]
		bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'h3', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
			{'name': 'x1', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x2', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x3', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x4', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x5', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x6', 'type': 'continuous', 'domain': (-2, 2)},
			{'name': 'x7', 'type': 'continuous', 'domain': (-2, 2)}]
	#/Laurens
	else:
		raise NotImplementedError

	# Run CoCaBO Algorithm
	if batch == 1:
		# sequential CoCaBO
		mabbo = CoCaBO(objfn=f, initN=initN, bounds=bounds,
					   acq_type='LCB', C=categories,
					   kernel_mix = kernel_mix)

	else:
		# batch CoCaBO
		mabbo = BatchCoCaBO(objfn=f, initN=initN, bounds=bounds,
							acq_type='LCB', C=categories,
							kernel_mix=kernel_mix,
							batch_size=batch)
	mabbo.runTrials(trials, budget, saving_path)




def read_cocabo(folder, num_runs,num_iters):
	cocabodata = []
	for i in range(num_runs):

		filename = folder + 'CoCaBO_1_best_vals_LCB_ARD_False_mix_0.5_df_s' + str(i)
		f = open(filename,'rb')
		cl = pickle.load(f)
		cocabodata.append(cl.best_value)
		#outname = 'run' + str(i) + '.xlsx'
		#cl.to_excel(outname)
	filename = folder + 'Cocabo_timeperiteration.txt'
	with open(filename, 'r') as f:
		Ctimes = f.readlines()
		Ctimes = np.copy(Ctimes[0:num_iters*num_runs+1])
		print(Ctimes[0])
		Ctimes = Ctimes.astype(float)
	print(Ctimes.shape)
	Ctimes = Ctimes.reshape((num_runs,num_iters))
	return cocabodata, Ctimes



# Read data from log file (this reads the best found objective values at each iteration)
def read_logs(folder):
	#folder = 'MVDONE/'
	allfiles = os.listdir(folder)
	logfiles = [f for f in allfiles if '.log' in f]
	allbests = []
	MVtimes = []
	for log in logfiles:
		with open('' + folder + log,'r') as f:
			MVDONEfile = f.readlines()
			MVDONE_best = []
			MVDONE_time = []
			for i, lines in enumerate(MVDONEfile):
				searchterm = 'Best data point according to the model and predicted value'
				if searchterm in lines:
					#print('Hello', MVDONEfile)
					temp = MVDONEfile[i-1]
					temp = temp.split('] , ')
					temp = temp[1]
					MVDONE_best.append(float(temp))
				searchterm2 = 'Total computation time for this iteration:'
				if searchterm2 in lines:
					#print('Hello', MVDONEfile)
					temp = MVDONEfile[i]
					temp = temp.split(':')
					temp = temp[1]
					MVDONE_time.append(float(temp))
		allbests.append(MVDONE_best)
		# print(np.copy(allbests))
		# print(np.copy(allbests).shape)
		# exit()
		MVtimes.append(MVDONE_time)
	return np.copy(allbests), np.copy(MVtimes)
	
	
# Plot the best found objective values at each iteration
def plot_results(folderCoCaBO, folderMVDONE):
	import matplotlib.pyplot as plt
	MVDONE_ev, MVtimes=read_logs(folderMVDONE)
	#print(MVtimes.shape)
	rand_iters = rand_evals
	total_iters = MVtimes.shape[1]
	#print(total_iters)
	#print(MVDONE_ev[0])
	avs_M = -np.mean(MVDONE_ev,0)
	avs_Mtime = np.mean(MVtimes,0)
	stds_M = np.std(MVDONE_ev,0)
	stds_Mtime = np.std(MVtimes,0)
	
	cocabodata, ctimes = read_cocabo(folderCoCaBO,n_trials,n_itrs)
	avs_C = np.mean(cocabodata,0)
	
	stds_C = np.std(cocabodata,0)
	avs_Ctime = np.mean(ctimes,0)
	stds_Ctime = np.std(ctimes,0)
	C_iters = len(avs_C)-1
	#print(len(avs_C), len(avs_M), len(avs_Ctime))
	#print(avs_Ctime[np.arange(0,C_iters,1)])
	#print('hoi', avs_Ctime.shape)
	
	plt.subplot(121)
	plt.errorbar(range(0,total_iters-rand_iters,1), avs_M[np.arange(rand_iters,total_iters,1)], yerr=stds_M[np.arange(rand_iters,total_iters,1)], errorevery=50, markevery=50, linestyle='-', linewidth=2.0, marker='s', capsize=5, label='MVDONE')
	plt.errorbar(range(1,C_iters,1), avs_C[np.arange(1,C_iters,1)], yerr=stds_C[np.arange(1,C_iters,1)], errorevery=50, markevery=50, linestyle='-', linewidth=2.0, marker='^', capsize=5, label='CoCaBO')
	plt.xlabel('Iteration')
	plt.ylabel('Objective')
	#plt.ylim((-20,0))
	plt.grid()
	plt.legend()
	
	plt.subplot(122)
	plt.errorbar(range(0,total_iters-rand_iters,1), avs_Mtime[np.arange(rand_iters,total_iters,1)], yerr=stds_Mtime[np.arange(rand_iters,total_iters,1)], errorevery=50, markevery=50, linestyle='-', linewidth=2.0, marker='s', capsize=5, label='MVDONE')
	plt.errorbar(range(0,C_iters,1), avs_Ctime[np.arange(0,C_iters,1)], yerr=stds_Ctime[np.arange(0,C_iters,1)], errorevery=50, markevery=50, linestyle='-', linewidth=2.0, marker='^', capsize=5, label='CoCaBO')
	plt.xlabel('Iteration')
	plt.ylabel('Computation time per iteration [s]')
	plt.yscale('log')
	#plt.ylim((1e-1,1e2))
	plt.grid()
	plt.legend()
	leg = plt.legend()
	if leg:
		leg.set_draggable(True)
	plt.show()
	






if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
	parser.add_argument('-f', '--func', help='Objective function',
						default='dim10Rosenbrock', type=str)
	parser.add_argument('-mix', '--kernel_mix',
						help='Mixture weight for production and summation kernel. Default = 0.0', default=0.5,
						type=float)
	parser.add_argument('-n', '--max_itr', help='Max Optimisation iterations. Default = 100',
						default=100, type=int)
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
	
	

	CoCaBO_Exps(obj_func=obj_func, budget=n_itrs,trials=n_trials, kernel_mix = kernel_mix, batch=batch)
	
	folder = '.\data\syntheticFns\dim10Rosenbrock\\'
	
	if obj_func == 'dim10Rosenbrock':
		ff = testFunctions.syntheticFunctions.dim10Rosenbrock
	
	d = 10 # Total number of variables
	lb = -2*np.ones(d).astype(int) # Lower bound
	ub = 2*np.ones(d).astype(int) # Upper bound
	num_int = 3 # number of integer variables
	lb[0:num_int] = 0
	ub[0:num_int] = num_int+1
	x0 =np.zeros(d)
	x0[0:num_int] = np.round(np.random.rand(num_int)*(ub[0:num_int]-lb[0:num_int]) + lb[0:num_int]) # Random initial guess (integer)
	x0[num_int:d] = np.random.rand(d-num_int)*(ub[num_int:d]-lb[num_int:d]) + lb[num_int:d] # Random initial guess (continuous)
	rand_evals = 24 # Number of random iterations, same as initN above
	max_evals = n_itrs+rand_evals # Maximum number of MVDONE iterations
	def obj_MVDONE(x):
		print(x[0:num_int])
		h = np.copy(x[0:num_int]).astype(int)
		return ff(h,x[num_int:])
	def run_MVDONE():
		solX, solY, model, logfile = MVDONE.MVDONE_minimize(obj_MVDONE, x0, lb, ub, num_int, max_evals, rand_evals)
		os.rename(logfile, folder+logfile)
		print("Solution found: ")
		print(f"X = {solX}")
		print(f"Y = {solY}")
	for i in range(n_trials):
		print(f"Testing MVDONE on the {d}-dimensional Rosenbrock function with integer constraints.")
		print("The known global minimum is f(1,1,...,1)=0")
		run_MVDONE()
		
	
	
	plot_results(folder, folder)
		

# Visualise the results
#MVDONE.plot_results(logfile)
#input('...')