#!/usr/bin/env python
# module EDMtools
import numpy as np

def classicalMDS(EDM_noisy, d):
	N = EDM_noisy.shape[0]
	oneN = np.ones(N)
	J = np.eye(N) - np.outer(oneN,oneN)/N
	G = -0.5*J@EDM_noisy@J
	eigVals, eigVecs = np.linalg.eig(G)
	tmp = np.diag(np.sqrt(eigVals[:d]))
	return np.real(np.hstack((tmp,np.zeros([d, N-d])))@eigVecs.T)

def procrustes(A, X):
	# columns of A are the anchors
	# columns of X are the points to be aligned to the anchors
	d = A.shape[0]
	L = A.shape[1]
	assert L>= d, 'Procrustes needs at least d anchors.'

	a_bar = np.reshape(np.mean(A, axis=1),(2,1))
	A_bar = A - a_bar
	x_bar = np.reshape(np.mean(X, axis=1),(2,1))
	X_bar = X - x_bar

	U, s, VT = np.linalg.svd(X_bar@A_bar.T, full_matrices=True)
	R = VT.T@U.T
	oneL = np.ones(L)
	return R@X_bar+a_bar

def semidefinite_relaxation(edm, W, lamda, print_out=False):
	import cvxpy as cvx
	def kappa(gram):
		n = len(gram)
		e = np.ones(n)
		return np.outer(np.diag(gram), e) + np.outer(e, np.diag(gram).T) - 2 * gram

	def kappa_cvx(gram, n):
		e = np.ones((n, 1))
		return cvx.diag(gram) * e.T + e * cvx.diag(gram).T - 2 * gram

	n = edm.shape[0]
	V = np.c_[-np.ones(n - 1) / np.sqrt(n), np.eye(n - 1) - np.ones([n - 1, n - 1]) / (n + np.sqrt(n))].T

	# Creates a n-1 by n-1 positive semidefinite variable.
	H = cvx.Semidef(n - 1)
	G = V * H * V.T  # * is overloaded
	edm_optimize = kappa_cvx(G, n)

	obj = cvx.Maximize(cvx.trace(H) - lamda * cvx.norm(cvx.mul_elemwise(W, (edm_optimize - edm))))
	prob = cvx.Problem(obj)

	## Solution
	total = prob.solve()
	if (print_out):
		print('total cost:', total)

	Gbest = V * H.value * V.T
	edm_complete = kappa(Gbest)

	# TODO why do these two not sum up to the objective?
	if (print_out):
		print('trace of H:', np.trace(H.value))
		print('other cost:', lamda * cvx.norm(cvx.mul_elemwise(W, (edm_complete - edm))).value)

	return edm_complete

def rank_alternation(edm_missing, W, d, niter=50, print_out=False, edm_true=None):
	def low_rank_approximation(A, r):
		U, s, VT = np.linalg.svd(A, full_matrices=False)
		s[r:] = 0
		return U@np.diag(s)@VT

	errs = []
	N = edm_missing.shape[0]
	edm_complete = edm_missing.copy()
	edm_complete[W != 1] = np.mean(edm_complete[W == 1])
	for i in range(niter):
		# impose matrix rank
		edm_complete = low_rank_approximation(edm_complete, d+2)

		# impose known entries
		edm_complete[W == 1] = edm_missing[W == 1]

		# impose matrix structure
		edm_complete[range(N), range(N)] = 0.0
		edm_complete[edm_complete < 0] = 0.0
		edm_complete = 0.5 * (edm_complete + edm_complete.T)

		if edm_true is not None:
			err = np.linalg.norm(edm_complete - edm_true)
			errs.append(err)

	return edm_complete, errs


if __name__ == "__main__":
	print('nothing happens when running this module.')
