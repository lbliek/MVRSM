# MVRSM on 10-dimensional Rosenbrock example
# By Laurens Bliek, 16-03-2020


import sys
# sys.path.append('../bayesopt')
# sys.path.append('../ml_utils')
import argparse
import os
import numpy as np
import pickle
import time
import testFunctions.syntheticFunctions
from methods.CoCaBO import CoCaBO
from methods.BatchCoCaBO import BatchCoCaBO
import MVRSM
from hyperopt import fmin, tpe, rand, hp, STATUS_OK, Trials
from functools import partial

from scipy.optimize import rosen


folder = ".\\data\\syntheticFns\\dim10Rosenbrock"
rand_evals = 24
n_itrs = 10
n_trials = 1
max_evals = rand_evals+n_itrs


def read_cocabo(folder, num_runs,num_iters):
	cocabodata = []
	for i in range(num_runs):

		filename = os.path.join(folder,'CoCaBO_1_best_vals_LCB_ARD_False_mix_0.5_df_s' + str(i))
		f = open(filename,'rb')
		cl = pickle.load(f)
		cocabodata.append(cl.best_value)
		#outname = 'run' + str(i) + '.xlsx'
		#cl.to_excel(outname)
	filename = os.path.join(folder, 'Cocabo_timeperiteration.txt')
	with open(filename, 'r') as f:
		Ctimes = f.readlines()
		Ctimes = np.copy(Ctimes[0:num_iters*num_runs])
		Ctimes = Ctimes.astype(float)
	#print(Ctimes.shape)
	Ctimes = Ctimes.reshape((num_runs,num_iters))
	return cocabodata, Ctimes



# Read data from log file (this reads the best found objective values at each iteration)
def read_logs_MVRSM(folder):
	#folder = 'MVRSM/'
	allfiles = os.listdir(folder)
	logfilesMV = [f for f in allfiles if ('.log' in f and 'MVRSM' in f)]
	MVbests = []
	MVtimes = []
	for log in logfilesMV:
		with open(os.path.join(folder,log),'r') as f:
			MVRSMfile = f.readlines()
			MVRSM_best = []
			MVRSM_time = []
			for i, lines in enumerate(MVRSMfile):
				searchterm = 'Best data point according to the model and predicted value'
				if searchterm in lines:
					#print('Hello', MVRSMfile)
					temp = MVRSMfile[i-1]
					temp = temp.split('] , ')
					temp = temp[1].strip()
					temp = temp.strip('[')
					temp = temp.strip(']')
					MVRSM_best.append(float(temp))
				searchterm2 = 'Total computation time for this iteration:	 '
				if searchterm2 in lines:
					#print('Hello', MVRSMfile)
					temp = MVRSMfile[i]
					temp = temp.split(':')
					temp = temp[1]
					if temp[0]:
						MVRSM_time.append(float(temp))
		MVbests.append(MVRSM_best)
		# print(np.copy(allbests))
		# print(np.copy(allbests).shape)
		# exit()
		MVtimes.append(MVRSM_time)
	return np.copy(MVbests), np.copy(MVtimes)
	
	
# Read data from log file (this reads the best found objective values at each iteration)
def read_logs_HO(folder, num_runs,num_iters):
	allfiles = os.listdir(folder)
	logfilesHO = [f for f in allfiles if ('.log' in f and 'HypOpt' in f)]
	HObests = []
	for log in logfilesHO:
		with open(os.path.join(folder,log),'r') as f:
			best = 10e9
			HOfile = f.readlines()
			HOfile = HOfile[0]
			HOfile = HOfile.split(',')
			HO_ev = []
			for i, lines in enumerate(HOfile):
				searchterm1 = "'result': {'loss': "
				if searchterm1 in lines:
					temp1 = lines
					temp1 = temp1.split(searchterm1)
					temp1 = temp1[1]
					temp1 = float(temp1)
					if temp1 < best:
						best = temp1
						HO_ev.append(temp1)
					else:
						HO_ev.append(best)
		HObests.append(HO_ev)
						
	filename = os.path.join(folder, 'HO_timeperiteration.txt')
	with open(filename, 'r') as f:
		HOtimes = f.readlines()
		HOtimes = np.copy(HOtimes[0:num_iters*num_runs])
		HOtimes = HOtimes.astype(float)
	HOtimes = HOtimes.reshape((num_runs,num_iters))
	return HObests, HOtimes
	
def read_logs_RS(folder, num_runs,num_iters):
	allfiles = os.listdir(folder)
	logfilesRS = [f for f in allfiles if ('.log' in f and 'RS_' in f)]
	RSbests = []
	for log in logfilesRS:
		with open(os.path.join(folder,log),'r') as f:
			best = 10e9
			RSfile = f.readlines()
			RSfile = RSfile[0]
			RSfile = RSfile.split(',')
			RS_ev = []
			for i, lines in enumerate(RSfile):
				searchterm1 = "'result': {'loss': "
				if searchterm1 in lines:
					temp1 = lines
					temp1 = temp1.split(searchterm1)
					temp1 = temp1[1]
					temp1 = float(temp1)
					if temp1 < best:
						best = temp1
						RS_ev.append(temp1)
					else:
						RS_ev.append(best)
		RSbests.append(RS_ev)
						
	filename = os.path.join(folder, 'RS_timeperiteration.txt')
	with open(filename, 'r') as f:
		RStimes = f.readlines()
		RStimes = np.copy(RStimes[0:num_iters*num_runs])
		RStimes = RStimes.astype(float)
	RStimes = RStimes.reshape((num_runs,num_iters))
	return RSbests, RStimes
	
	
# Plot the best found objective values at each iteration
def plot_results(folderCoCaBO, folderMVRSM, folderHO, folderRS, rand_evals=rand_evals, n_itrs=n_itrs, n_trials=n_trials):
	import matplotlib.pyplot as plt
	MVRSM_ev, MVtimes=read_logs_MVRSM(folderMVRSM)
	MVRSM_ev = MVRSM_ev.astype(float)
	MVtimes = MVtimes.astype(float)
	
	rand_iters = rand_evals
	total_iters = max_evals
	avs_M = -np.mean(MVRSM_ev,0)
	avs_Mtime = np.mean(MVtimes,0)
	stds_M = np.std(MVRSM_ev,0)
	stds_Mtime = np.std(MVtimes,0)
	
	
	HO_ev, HOtimes = read_logs_HO(folderHO,n_trials,total_iters)
	#print(HOtimes.shape)
	avs_HO = -np.mean(HO_ev,0)
	avs_HOtime = np.mean(HOtimes,0)
	stds_HO = np.std(HO_ev,0)
	stds_HOtime = np.std(HOtimes,0)
	
	
	
	RS_ev, RStimes = read_logs_RS(folderRS,n_trials,total_iters)
	avs_RS = -np.mean(RS_ev,0)
	avs_RStime = np.mean(RStimes,0)
	stds_RS = np.std(RS_ev,0)
	stds_RStime = np.std(RStimes,0)
	
	#print(MVtimes.shape)
	
	Rosenbrock238=False #don't plot CoCaBO for Rosenbrock238
	
	if not Rosenbrock238:
		cocabodata, ctimes = read_cocabo(folderCoCaBO,n_trials,n_itrs)
		avs_C = np.mean(cocabodata,0)	
		stds_C = np.std(cocabodata,0)
		avs_Ctime = np.mean(ctimes,0)
		stds_Ctime = np.std(ctimes,0)
		#C_iters = len(avs_C)-1
		C_iters = n_itrs
		#print(len(avs_C), len(avs_M), len(avs_Ctime))
		#print(avs_Ctime[np.arange(0,C_iters,1)])
		#print('hoi', avs_Ctime.shape)
	
	print("RS total time: ", np.sum(avs_RStime), " +- ", np.sum(stds_RStime))
	print("HO total time: ", np.sum(avs_HOtime), " +- ", np.sum(stds_HOtime))
	print("MVRSM total time: ", np.sum(avs_Mtime), " +- ", np.sum(stds_Mtime))
	if not Rosenbrock238:
		print("COCABO total time: ", np.sum(avs_Ctime), " +- ", np.sum(stds_Ctime))
	
	#print(avs_HOtime)
	
	plt.figure(figsize=(7,3.5))
	
	errorevery = int(n_itrs/10)
	markevery = int(n_itrs/10)
	plt.subplot(121)
	plt.subplots_adjust(left=0.11, bottom=0.16, right=0.96, top=0.90, wspace=0.41, hspace=0.2)
	plt.errorbar(range(0,n_itrs,1), avs_RS[np.arange(rand_iters,total_iters,1)], yerr=stds_RS[np.arange(rand_iters,total_iters,1)], errorevery=errorevery, markevery=markevery, linestyle='-', linewidth=2.0, marker='o', capsize=5, label='RS')
	plt.errorbar(range(0,n_itrs,1), avs_HO[np.arange(rand_iters,total_iters,1)], yerr=stds_HO[np.arange(rand_iters,total_iters,1)], errorevery=errorevery, markevery=markevery, linestyle='-', linewidth=2.0, marker='d', capsize=5, label='HO')
	plt.errorbar(range(0,n_itrs,1), avs_M[np.arange(rand_iters,total_iters,1)], yerr=stds_M[np.arange(rand_iters,total_iters,1)], errorevery=errorevery, markevery=markevery, linestyle='-', linewidth=2.0, marker='s', capsize=5, label='MVRSM')
	if not Rosenbrock238:
		plt.errorbar(range(0,n_itrs,1), avs_C[np.arange(0,n_itrs,1)], yerr=stds_C[np.arange(0,n_itrs,1)], errorevery=errorevery, markevery=markevery, linestyle='-', linewidth=2.0, marker='^', capsize=5, label='CoCaBO')
	plt.xlabel('Iteration')
	plt.ylabel('Objective')
	#plt.ylim((-20,0))
	plt.grid()
	leg = plt.legend()
	if leg:
		leg.set_draggable(True)
	#plt.show()
	
	
	plt.subplot(122)
	plt.errorbar(range(0,n_itrs,1), avs_RStime[np.arange(rand_iters,total_iters,1)], yerr=stds_RStime[np.arange(rand_iters,total_iters,1)], errorevery=errorevery, markevery=markevery, linestyle='-', linewidth=2.0, marker='o', capsize=5, label='RS')
	plt.errorbar(range(0,n_itrs,1), avs_HOtime[np.arange(rand_iters,total_iters,1)], yerr=stds_HOtime[np.arange(rand_iters,total_iters,1)], errorevery=errorevery, markevery=markevery, linestyle='-', linewidth=2.0, marker='d', capsize=5, label='HO')
	plt.errorbar(range(0,n_itrs,1), avs_Mtime[np.arange(rand_iters,total_iters,1)], yerr=stds_Mtime[np.arange(rand_iters,total_iters,1)], errorevery=errorevery, markevery=markevery, linestyle='-', linewidth=2.0, marker='s', capsize=5, label='MVRSM')
	if not Rosenbrock238:
		plt.errorbar(range(0,n_itrs,1), avs_Ctime[np.arange(0,n_itrs,1)], yerr=stds_Ctime[np.arange(0,n_itrs,1)], errorevery=errorevery, markevery=markevery, linestyle='-', linewidth=2.0, marker='^', capsize=5, label='CoCaBO')
	plt.xlabel('Iteration')
	plt.ylabel('Computation time per iteration [s]')
	plt.yscale('log')
	#plt.ylim((1e-4,1e4)) #only needed for Ackley53
	plt.grid()
	plt.legend()
	leg = plt.legend()
	if leg:
		leg.set_draggable(True)
	plt.show()
	



plot_results(folder, folder, folder, folder, rand_evals=rand_evals, n_itrs=n_itrs, n_trials=n_trials)