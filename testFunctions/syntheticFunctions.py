# Adapted from:
#
# -*- coding: utf-8 -*-
#==========================================
# Title:  syntheticFunctions.py
# Author: Binxin Ru and Ahsan Alvi
# Date:	  20 August 2019
# Link:	  https://arxiv.org/abs/1906.08878
#==========================================

import numpy as np
from scipy.optimize import rosen
import math

# =============================================================================
# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X):
	X = np.asarray(X)
	X = X.reshape((-1, 2))
	if len(X.shape) == 1:  # one observation
		x1 = X[0]
		x2 = X[1]
	else:  # multiple observations
		x1 = X[:, 0]
		x2 = X[:, 1]
	fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
	return fx.reshape(-1, 1) / 300

#Adapted
def highdimRosenbrock(ht_list,x):
		XX = []
		assert len(ht_list) == 5
		h2 = [-2, -1, 0, 1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
		for i in ht_list:
			if i:
				XX.append(h2[i])
			else:
				XX.append(h2[0])
		for i in x:
			XX.append(i)
		XX[0:len(ht_list)]=np.round(XX[0:len(ht_list)]) #To make sure there is no cheating, round the discrete variables before calling the function
		return rosen(XX)/300 + 1e-6 * np.random.rand()

def dim10Rosenbrock(ht_list,x):
		XX = []
		assert len(ht_list) == 3
		h2 = [-2, -1, 0, 1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
		for i in ht_list:
			if i:
				XX.append(h2[i])
			else:
				XX.append(h2[0])
		for i in x:
			XX.append(i)
		XX[0:len(ht_list)]=np.round(XX[0:len(ht_list)]) #To make sure there is no cheating, round the discrete variables before calling the function
		return rosen(XX)/300 + 1e-6 * np.random.rand()

def dim53Rosenbrock(ht_list,x):
		XX = []
		assert len(ht_list) == 50
		h2 = [1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
		for i in ht_list:
			if i:
				XX.append(h2[i])
			else:
				XX.append(h2[0])
		for i in x:
			XX.append(i)
		XX[0:len(ht_list)]=np.round(XX[0:len(ht_list)]) #To make sure there is no cheating, round the discrete variables before calling the function
		return rosen(XX)/20000 + 1e-6 * np.random.rand()
		
def dim53Ackley(ht_list,x):
		XX = []
		assert len(ht_list) == 50
		h2 = [0, 1] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
		
		for i in ht_list:
			if i:
				XX.append(h2[i])
			else:
				XX.append(h2[0])
		for i in x:
			XX.append(i)
		XX[0:len(ht_list)]=np.round(XX[0:len(ht_list)]) #To make sure there is no cheating, round the discrete variables before calling the function
		a = 20
		b = 0.2
		c = 2*np.pi
		sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(XX))/53))
		cos_term = -1*np.exp(np.sum(np.cos(c*np.copy(XX))/53))
		result = a + np.exp(1) + sum_sq_term + cos_term
		return result + 1e-6 * np.random.rand()
		
		
def dim238Rosenbrock(ht_list,x):
		XX = []
		assert len(ht_list) == 119
		h2 = [-2, -1, 0, 1, 2] #convert to these categories, as Cocabo assumes categories in (0,1,2,3,etc.)
		for i in ht_list:
			if i:
				XX.append(h2[i])
			else:
				XX.append(h2[0])
		for i in x:
			XX.append(i)
		XX[0:len(ht_list)]=np.round(XX[0:len(ht_list)]) #To make sure there is no cheating, round the discrete variables before calling the function
		return rosen(XX)/50000 + 1e-6 * np.random.rand()
#/Adapted


def Kursawe1(x):
	f = -10*math.exp(-0.2*math.sqrt(x[0]**2+x[1]**2)) + -10*math.exp(-0.2*math.sqrt(x[1]**2+x[2]**2))
	return f
def Kursawe2(x):
	f = abs(x[0])**0.8 + 5*math.sin(x[0]**3) + abs(x[1])**0.8 + 5*math.sin(x[1]**3) + abs(x[2])**0.8 + 5*math.sin(x[2]**3)
	return f

def ZDT3_1(x):
	f = x[0]
	return f
def ZDT3_2(x):
	g = 1+9/29*sum(x[1:30])
	h = 1-math.sqrt(x[0]/g) - (x[0]/g)*math.sin(10*3.1415926535*x[0])
	f = g*h
	return f
def ZDT6_1(x):
	#from pymoo
	f1 = 1 - math.exp(-4 * x[0]) * np.power(math.sin(6 * 3.1415926535 * x[0]), 6)
	#g = 1 + 9.0 * math.power(anp.sum(x[:, 1:], axis=1) / (self.n_var - 1.0), 0.25)
	#f2 = g * (1 - math.power(f1 / g, 2))
	return f1
def ZDT6_2(x):
	#from pymoo
	f1 = 1 - math.exp(-4 * x[0]) * np.power(math.sin(6 * 3.1415926535 * x[0]), 6)
	g = 1 + 9.0 * np.power(np.sum(x[1:]) / (10 - 1.0), 0.25)
	f2 = g * (1 - np.power(f1 / g, 2))
	return f2
def Poloni_1(x):
	A1 = 0.5*math.sin(1) -2*math.cos(1)+ math.sin(2) - 1.5*math.cos(2)
	A2 = 1.5 * math.sin(1) - math.cos(1) + 2*math.sin(2) - 0.5 * math.cos(2)
	B1 = 0.5 * math.sin(x[0]) - 2 * math.cos(x[0]) + math.sin(x[1]) - 1.5 * math.cos(x[1])
	B2 = 1.5 * math.sin(x[0]) - math.cos(x[0]) + 2*math.sin(x[1]) - 0.5 * math.cos(x[1])
	f1 = 1+(A1-B1)**2 +(A2-B2)**2
	return f1
def Poloni_2(x):
	f2 = (x[0]+3)**2 + (x[1]+1)**2
	return f2
def Schaffer2_1(x):
	if x[0] <= 1:
		f1 = -x[0]
	elif x[0]<=3:
		f1 = x[0]-2
	elif x[0]<=4:
		f1 = 4-x[0]
	elif x[0]>4:
		f1 = x[0]-4
	else:
		print('Warning: no value for x found')
	return f1
def Schaffer2_2(x):
	f2 = (x[0]-5)**2
	return f2
def SymPart_1(x):
	X1 = x[0]
	X2 = x[1]
	a = 1
	b = 10
	c = 10
	w = np.pi/4
	t1_hat = np.sign(X1) * np.ceil((np.abs(X1) - a - c / 2) / (2 * a + c))
	t2_hat = np.sign(X2) * np.ceil((np.abs(X2) - b / 2) / b)
	one = 1 #np.ones(len(X))
	t1 = np.sign(t1_hat) * np.min(np.vstack((np.abs(t1_hat), one)), axis=0)
	t2 = np.sign(t2_hat) * np.min(np.vstack((np.abs(t2_hat), one)), axis=0)
	p1 = X1 - t1 * c
	p2 = X2 - t2 * b

	f1 = (p1 + a) ** 2 + p2 ** 2
	f2 = (p1 - a) ** 2 + p2 ** 2
	return f1[0]
def SymPart_2(x):
	X1 = x[0]
	X2 = x[1]
	a = 1
	b = 10
	c = 10
	w = np.pi/4
	t1_hat = np.sign(X1) * np.ceil((np.abs(X1) - a - c / 2) / (2 * a + c))
	t2_hat = np.sign(X2) * np.ceil((np.abs(X2) - b / 2) / b)
	one = 1 #np.ones(len(X))
	t1 = np.sign(t1_hat) * np.min(np.vstack((np.abs(t1_hat), one)), axis=0)
	t2 = np.sign(t2_hat) * np.min(np.vstack((np.abs(t2_hat), one)), axis=0)
	p1 = X1 - t1 * c
	p2 = X2 - t2 * b

	f1 = (p1 + a) ** 2 + p2 ** 2
	f2 = (p1 - a) ** 2 + p2 ** 2
	return f2[0]


# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html		  
# =============================================================================
def mysixhumpcamp(X):
	X = np.asarray(X)
	X = np.reshape(X, (-1, 2))
	if len(X.shape) == 1:
		x1 = X[0]
		x2 = X[1]
	else:
		x1 = X[:, 0]
		x2 = X[:, 1]
	term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
	term2 = x1 * x2
	term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
	fval = term1 + term2 + term3
	return fval.reshape(-1, 1) / 10

# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X):
	X = np.asarray(X) / 2
	X = X.reshape((-1, 2))
	if len(X.shape) == 1:
		x1 = X[0] * 2
		x2 = X[1] * 2
	else:
		x1 = X[:, 0] * 2
		x2 = X[:, 1] * 2
	fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
			2.625 - x1 + x1 * x2 ** 3) ** 2
	return fval.reshape(-1, 1) / 50


def func2C(ht_list, X):
	# ht is a categorical index
	# X is a continuous variable
	X = X * 2

	assert len(ht_list) == 2
	ht1 = ht_list[0]
	ht2 = ht_list[1]

	if ht1 == 0:  # rosenbrock
		f = myrosenbrock(X)
	elif ht1 == 1:	# six hump
		f = mysixhumpcamp(X)
	elif ht1 == 2:	# beale
		f = mybeale(X)

	if ht2 == 0:  # rosenbrock
		f = f + myrosenbrock(X)
	elif ht2 == 1:	# six hump
		f = f + mysixhumpcamp(X)
	else:
		f = f + mybeale(X)

	y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])
	return y.astype(float)


def func3C(ht_list, X):
	# ht is a categorical index
	# X is a continuous variable
	X = np.atleast_2d(X)
	assert len(ht_list) == 3
	ht1 = ht_list[0]
	ht2 = ht_list[1]
	ht3 = ht_list[2]

	X = X * 2
	if ht1 == 0:  # rosenbrock
		f = myrosenbrock(X)
	elif ht1 == 1:	# six hump
		f = mysixhumpcamp(X)
	elif ht1 == 2:	# beale
		f = mybeale(X)

	if ht2 == 0:  # rosenbrock
		f = f + myrosenbrock(X)
	elif ht2 == 1:	# six hump
		f = f + mysixhumpcamp(X)
	else:
		f = f + mybeale(X)

	if ht3 == 0:  # rosenbrock
		f = f + 5 * mysixhumpcamp(X)
	elif ht3 == 1:	# six hump
		f = f + 2 * myrosenbrock(X)
	else:
		f = f + ht3 * mybeale(X)

	y = f + 1e-6 * np.random.rand(f.shape[0], f.shape[1])

	return y.astype(float)
