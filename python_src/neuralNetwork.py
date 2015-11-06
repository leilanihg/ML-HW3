import numpy as np

class neuralNetwork():
	def __init__(self, x, M):
		""" 2 layer neural network """
		self.x = x
		self.M = M
		(self.K, self.D) = x.shape

		# Initialize sizes of the variables
		self.w1 = np.zeros([self.D, self.M])
		self.w2 = np.zeros([self.M + 1, self.K])
		self.z = np.zeros(self.M+1)
		self.y = np.zeros(self.K)


   	def F(self, x):
   		f = x[0]*x[0]+ x[1]*x[1]
   		return f

   	def grad(self, x):
   		return np.array([2*x[0], 2*x[1]])

   	def grad_approx(self, x, h):
   		n = len(x)
   		fin_dif = np.zeros([n])
   		for i in xrange(n):
			delta_vec = np.zeros([n])
			delta_vec[i] = .5*h
			approx = (1.0/h) * (self.F(x+delta_vec)-self.F(x-delta_vec))
			fin_dif[i] = approx
		return fin_dif

	def grad_descent(self, SSE=False, limit = 10):
		""" Run gradient descent on a scalar function """
		old = self.first
		if SSE:
			new = self.SSE_step(old)
		else:
			new = self.step(old)
		num_steps = 0 								# To keep track of how many iterations
		while(not conv_criteria(old, new, self.eps)):
			old = new
			if SSE:
				new = self.SSE_step(old)
			else:
				new = self.step(old)
			num_steps+=1
			if self.verbose:
				print "step ", num_steps, " old ", old, " new ", new
			#if num_steps >= 10000:
			#	break
		print "STEPS ", num_steps
		return new


	def step(self, old):
		if self.verbose:
			print "     GRADIENT FOR STEP ",  self.grad(old)
		new = old - self.step_size * self.grad(old)
		return new

	def SSE_step(self, old):
		#if self.verbose:
		#	print "     GRADIENT FOR STEP ",  hw1.computeSSEGrad(self.X, self.Y, old, self.order).T
		new = old - self.step_size * hw1.computeSSEGrad(self.X, self.Y, old, self.order).T
		return new

def grad_descent_LAD(X,Y, old, order, lam, h, limit = 10):
	""" Run gradient descent for LAD  """
	new = LAD_step(X, Y, old, order, lam, h, 0)
	num_steps = 0 								# To keep track of how many iterations
	while(not conv_criteria(old, new, .000001)):
		old = new
		new = LAD_step(X,Y, old, order, lam, h, num_steps)
		num_steps+=1
		if num_steps >= 10000:
			break
	return new

def LAD_step(X,Y, old, order, lam, h, num_steps):
	if(num_steps>=1):
		dec_step = (.1)*1.0/(np.sqrt(num_steps))
	else:
		dec_step = .1
	new = old - dec_step * gradient_approx_LAD(X, Y, old, order, lam, h).T
	return new

def gradient_approx_SSE(X, Y, weights, order, h):
	""" Calculates the gradient using finite differences """
	n = len(weights)
	diff = np.zeros([n,1])
	weight = weights.flatten()
	for i in xrange(n):
		delta_vec = np.zeros([n,1])
		delta_vec[i] = .5*h
		diff[i] = (1.0/h) * (f(X, Y, (weights+delta_vec), order) - f(X, Y, (weights-delta_vec), order))
	return diff.flatten()

def gradient_approx_LAD(X, Y, weights, order, lam, h):
	""" Calculates the gradient using finite differences """
	n = len(weights)
	diff = np.zeros([n,1])
	weight = weights.flatten()
	for i in xrange(n):
		delta_vec = np.zeros([n,1])
		delta_vec[i] = .5*h
		diff[i] = (1.0/h) * (hw1.computeLAD(X,Y, (weights+delta_vec), order, lam) - \
			hw1.computeLAD(X, Y, (weights-delta_vec), order, lam))
	return diff.flatten()

def f(X, Y, weights, order):
	""" define this yourself """
	return hw1.computeSSE(X, Y, weights, order)

def conv_criteria(current, previous, eps):
	""" Determines whether the algorithm has converged by the two-norm """
	diff = np.linalg.norm(current-previous, 2)
	if diff <= eps:
		return True
	else:
		return False	

if __name__ == '__main__':
	M = 1
	#x = np.ones(M+1)
	x = np.array([.8,-1])
	#x = np.array([.3, 8, -20, 17])
	des_obj = gradDesc(x, .01, .00001, True, 'curvefitting.txt', M)
	answer = des_obj.grad_descent(True)
	print "Found root at ", answer
    
