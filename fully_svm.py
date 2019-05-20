import random
import numpy as np

class SVM(object):
	"""docstring for SVM"""
	def __init__(self, arg):
		self.C = C
		self.gramma = gramma
		self.tol = tol
		self.max_iter = max_iter
		self.kernel = kernel

	def calculate_fx(i,dataMatrix,labelMatrix,m,b,alphas):
		fx = 0
		for k in range(m):
			fx += alphas[k]*labelMatrix[k]*self.svm_kernel(self.kernel,dataMatrix[k],dataMatrix[i])
		return fx

	def calculate_error(i,dataMatrix,labelMatrix,m,b,alphas):
		fx = 0
		for j in range(m):
			#Error[i] += a[i]*y[i]*np.dot(X[i],X[i])
			fx += alphas[j]*labelMatrix[j]*self.svm_kernel(self.kernel,dataMatrix[j],dataMatrix[i])
		fx += b
		E = fx - labelMatrix[i]
		return E

	def take_step(i,j,dataMatrix,labelMatrix,m,b,alphas):
		E[j] = self.calculate_error(j,dataMatrix,labelMatrix,m,b,alphas)
		alphaIold = alphas[i].copy()
		alphaJold = alphas[j].copy()
		if labelMatrix[i] == labelMatrix[j]:
			L = max(0,alphas[i]+alphas[j]-self.C)
			H = min(self.C,alphas[i]+alphas[j])
		else:
			L = max(0,alphas[j]-alphas[i])
			H = min(self.C,self.C+alphas[j]-alphas[i])
		if L == H:
			print("L == H")
			return 0
		k11 = self.svm_kernel(self.kernel,dataMatrix[i],dataMatrix[i])
		k22 = self.svm_kernel(self.kernel,dataMatrix[j],dataMatrix[j])
		k12 = self.svm_kernel(self.kernel,dataMatrix[i],dataMatrix[j])
		eta = 2*k12 - k11 - k22
		if eta < 0:
			alphas[j] = alphas[j] - labelMatrix[j]*(E[i]-E[j])/eta
			if alphas[j] > H:
				alphas[j] = H
			elif alphas[j] < L:
				alphas[j] = L
		else:
			#eta >= 0 的情况，这种情况下需要计算目标函数W,具体的计算方法参看smo算法原理.pdf公式12.23
			s = labelMatrix[i]*labelMatrix[j]
			k21 = k12
			lv1 = self.calculate_fx(i) + b - labelMatrix[i]*alphas[i]*k11 - labelMatrix[j]*L*k21
			lv2 = self.calculate_fx(j) + b - labelMatrix[i]*alphas[i]*k12 - labelMatrix[j]*L*k22
			hv1 = self.calculate_fx(i) + b - labelMatrix[i]*alphas[i]*k11 - labelMatrix[j]*H*k21
			hv2 = self.calculate_fx(j) + b - labelMatrix[i]*alphas[i]*k12 - labelMatrix[j]*H*k22
			lgramma = alphas[i] + s * L
			hgramma = alphas[i] + s * H
			lobj = lgramma - s * L + L - 1/2*k11*(lgramma - s*L)**2 - 1/2*k22*L**2 - s*k12*(lgramma - s*L)*L - labelMatrix[i]*(lgramma - s*L)*lv1 - labelMatrix[j]*L*lv2
			hlobj = hgramma - s * H + H - 1/2*k11*(hgramma - s*H)**2 - 1/2*k22*H**2 - s*k12*(hgramma - s*H)*H - labelMatrix[i]*(hgramma - s*H)*hv1 - labelMatrix[j]*H*hv2
			if lobj > hlobj + tol:
				alphas[j] = L
			elif lobj < hlobj - tol :
				alphas[j] = H
		if alphas[j] < 1e-8:
			alphas[j] = 0
		if alphas[j] > self.C - 1e-8:
			alphas[j] = self.C
		if abs(alphaJold-alphas[j])<0.00001:
			return 0
		alphas[i] = alphas[i] + labelMatrix[i]*labelMatrix[j](alphaJold-alphas[j])
		b1 = b - E[i] - labelMatrix[i]*(alphas[i] - alphaIold)*k11 - labelMatrix[j]*(alphas[j] - alphaJold)*k12
		b2 = b - E[j] - labelMatrix[i]*(alphas[i] - alphaIold)*k12 - labelMatrix[j]*(alphas[j] - alphaJold)*k22
		if alphas[i] > 0 and alphas[i] < self.C：
			b = b1
		elif alphas[j] > 0 and alphas[j] < self.C：
			b = b2
		else:
			b = (b1 + b2)/2
		num_changed_alphas += 1
		if (num_changed_alphas == 0): 
			passes += 1
		else: 
			passes = 0
		return 1

	def examine_example(i,dataMatrix,labelMatrix,m,b,alphas):
		E[i] = self.calculate_error(i,dataMatrix,labelMatrix,m,b,alphas)
		if (labelMatrix[i]*E[i] < -self.tol and alphas[i] < self.C) or \
			(labelMatrix[i]*E[i] > self.tol and alphas[i] > 0):
			non_boundIs = [i for i in range(m) if alphas[i] > 0 and alphas[i] < self.C]
			max_deltaE = 0
			j = -1
			if len(non_boundIs) >1:
				for k in non_boundIs:
					if k == i :continue
					E[k] = self.alculate_error(i,dataMatrix,labelMatrix,m,b)
					deltaE = abs(E[i]-E[k])
					if deltaE > max_deltaE:
						max_deltaE = deltaE
						j = k
				if self.take_step(i,j,dataMatrix,labelMatrix,m,b,alphas):
					return 1
			#随机打乱non_boundIs
			random_non_boundIs = non_boundIs.copy()
			random.shuffle(random_non_boundIs)
			for j in random_non_boundIs:
				if j == i :continue
				if self.take_step(i,j,dataMatrix,labelMatrix,m,b,alphas):
					return 1
			random_list = [i for i in range(m)]
			random.shuffle(random_list)
			for j in random_list:
				if j == i : continue
				if self.take_step(i,j,dataMatrix,labelMatrix,m,b,alphas):
					return 1
		return 0

	def fit(self ,data , classLabel, sample_weight=None):
		#完整的smo算法,初始化参数alphas,阈值
		dataMatrix = np.mat(data)
		labelMatrix = np.mat(classLabel)．T
		m,n = np.shape(dataMatrix)
		#alphas = np.mat(np.zeros(m,1))
		alphas = [0 for i in range(m)]
		b = 0
		num_changes_alphas = 0
		examine_all = 1
		while num_changes_alphas >0 or examine_all:
			num_changes_alphas = 0
			if examine_all:
				# loop I over all training example
				for i in range(m):
					num_changes_alphas += self.examine_example(i,dataMatrix,labelMatrix,m,b,alphas)
			else :
				#loop I over examples where alphas in not 0 and not C
				non_boundIs = [i for i in range(m) if alphas[i] > 0 and alphas[i] < self.C]
				#non_boundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
				for i in on_boundIs:
					num_changes_alphas += self.examine_example(i,dataMatrix,labelMatrix,m,b,alphas)
			if examine_all == 1:
				examine_all == 0
			elif num_changes_alphas == 0:
				examine_all == 1