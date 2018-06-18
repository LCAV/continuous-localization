#!/usr/bin/env python
# module SampTrajsTools

import numpy as np

def lowRankApproximation(A, r):
	U, s, VT = np.linalg.svd(A, full_matrices=False)
	s[r:] = 0
	return U@np.diag(s)@VT

def reconstructDTR(C, F, A):
	N = F.shape[1]
	M = A.shape[1]
	return np.outer(np.ones(N),np.diag(A.T@A))-2*F.T@C.T@A+np.outer(np.diag(F.T@C.T@C@F),np.ones(M))

def customMDS(DTR, F, A):
	[d, M] = A.shape
	N = F.shape[1]
	JM = np.eye(M)-np.ones([M,M])/M
	tmp = lowRankApproximation(JM@(np.outer(np.diag(A.T@A),np.ones(N)) - DTR.T), d)
	return 0.5*np.linalg.inv(A@JM@A.T)@A@tmp@F.T@np.linalg.inv(F@F.T)

def SRLS(A,F,C,DTR):
	DTR_hat = reconstructDTR(C, F, A)
	return np.linalg.norm(DTR-DTR_hat,2)

def getSRLSGrad(A,F,C,DTR):
	[K,N] = F.shape
	M = A.shape[1]
	At = A.transpose()
	Ft = F.transpose()
	Ct = C.transpose()
	LHS = A@(np.outer(np.diag(At@A),np.ones(N))-DTR.transpose())@Ft

	term1 =  M*np.outer(np.diag(Ft@Ct@C@F),np.ones(K))
	term2 = - 2*Ft@Ct@A@np.outer(np.ones(M),np.ones(K))
	term3 = - DTR@np.outer(np.ones(M),np.ones(K))
	RHS =  C@F@((term1+term2+term3)*Ft)+np.sum(np.diag(At@A))*C@F@Ft+A@(2*At@C@F-np.outer(np.ones(M),np.diag(Ft@Ct@C@F)))@Ft

	return RHS - LHS
    
def checkStationaryPointSRLS(A,F,C,DTR):
	return np.isclose(getSRLSGrad(A,F,C,DTR), 0)
    
def gradientStep(A,F,C,DTR,maxIters=10):
	grad = getSRLSGrad(A,F,C,DTR)
	bestCost = SRLS(A,F,C,DTR)
	C_hat = C
	minStep=0
	maxStep=0.01
	for i in range(maxIters):
		step = (maxStep-minStep)/2
		C_test = C_hat-step*grad
		cost = SRLS(A,F,C_test,DTR)
		if cost<bestCost:
			bestCost=cost
			C_hat = C_test
			minStep = step
			maxStep = 2*maxStep
		else:
			maxStep = step
			minStep = minStep/2
	return C_hat, bestCost

def gradientDescent(A,F,C,DTR,maxIters=10):
	C_hat = C
	costs = [SRLS(A,F,C,DTR)]
	for i in range(maxIters):
		C_hat, cost = gradientStep(A,F,C_hat,DTR,maxIters=100)
		costs.append(cost)
	return C_hat, costs


def alternateGDandKEonDR(DR_missing, W, F, A, niter=50, print_out=False, DR_true=None): 
	d = A.shape[0]
	N = F.shape[1]
	DR_complete = DR_missing.copy()
	DR_complete[W != 1] = np.mean(DR_complete[W == 1])

	#DR_complete[DR_complete < 0] = 0.0
	#C = customMDS(DR_complete[:N,:], F, A)
	#C, costs = gradientDescent(A,F,C,DR_complete[:N,:], maxIters=10)
	#DR_complete[:N,:] = reconstructDTR(C, F, A)
	
	if DR_true is not None:
		errs = [np.linalg.norm(DR_complete - DR_true)]
	else:
		errs = []

	for i in range(niter):
		# impose matrix rank
		#DR_complete = lowRankApproximation(DR_complete, d+2)

		# impose known entries
		DR_complete[W == 1] = DR_missing[W == 1]


		# zero negastive values
		DR_complete[DR_complete < 0] = 0.0

		# approximate coeffiecients
		C = customMDS(DR_complete[:N,:], F, A)
		C, costs = gradientDescent(A,F,C,DR_complete[:N,:], maxIters=10)

		# update DR
		DR_complete[:N,:] = reconstructDTR(C, F, A)

		if DR_true is not None:
			err = np.linalg.norm(DR_complete - DR_true)
			errs.append(err)

	return DR_complete, errs

if __name__ == "__main__":
	print('nothing happens when running this module.')