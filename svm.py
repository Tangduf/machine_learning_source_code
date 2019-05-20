#编写svm源码解决实际问题（先自己想，不懂再参看别人的代码）
#解决什么实际问题（元动作识别，ride,run,walk,sit）
#关键点是smo算法
#第一步是数据的读取和存储，然后初始化参数，为需要的变量分配空间，然后就是伪代码的实现

# read data and store data
# 1)样本数量不大，万级别　　2)巨大数量的样本，十万百万级别
# 样本数量不大，直接用pandas读取数据文件进入内存，然后对数据进行预处理，将数据划分为测试集和训练集



# 为svm模型建立一个类
import random

class SVM (object):
	"""SVM中常见参数Ｃ，gramma,核函数类型，tol,最大迭代次数max_iter

SVM """
	def __init__(self, C,gramma,tol,max_iter,kernel):

		self.C = C
		self.gramma = gramma
		self.tol = tol
		self.max_iter = max_iter
		self.kernel = kernel

	def calculate_error(self,dataMatrix,labelMatrix,i,b,m,alphas):
		for k in range(m):
			#Error[i] += a[i]*y[i]*np.dot(X[i],X[i])
			fx += alphas[k]*labelMatrix[k]*self.svm_kernel(self.kernel,dataMatrix[k],dataMatrix[i])
		fx += b
		E = fx - labelMatrix[i]
		return E

	def predict(self):
		return 0

	def svm_kernel(self,kernel,x,z):
		return 0

	def fit(self ,data , classLabel, sample_weight=None):
		""" Fit the model according to the given training data
		Parameters
		----------
		X:{array-like,sparse matrix},shape = [n_samples,n_features]
		y:array-like,shape = [n_samples]
		sample_weight:array_like,shape = [n_samples],optional
		Array of weights that are assigned to individual samples.If not provides,
		then each sample is given unit weight

		returns
		----------
		self :object
		"""
		#pandas用于统计分析(类似于表格),将数据转化为np矩阵，方便后续的处理
		dataMatrix = np.mat(data)
		labelMatrix = np.mat(classLabel)．T
		#获得样本的数目,初始化参数alphas,b参数
		m,n = np.shape(dataMatrix)
		b = 0
		alphas = [0 for i in range(m)]
		#初始化参数alphas,b
		#a = [0 for i in range(m)]
		#a = np.zeros(m)
		passes = 0
		#初始化E
		E = [0 for i in range(m)]

		while passes < max_passes:
			num_changed_alphas = 0
			# row 是 X　二维矩阵中的每一行
			for i in range(m):
				#计算Ei
				E[i] = self.calculate_error(dataMatrix,labelMatrix,i,self.b,m,alphas)
				#找到不满足ktt条件的样本
				if (labelMatrix[i]*E[i] < -self.tol and alphas[i] < self.C) or \
				 (labelMatrix[i]*E[i] > self.tol and alphas[i] > 0)
					#选择一个不等于i的随机数
					j = i
					while j == i:
						j = random.randint(0, m-1)
					E[j] = self.calculate_error(dataMatrix,labelMatrix,j,self.b,m,alphas)
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
						continue
					eta = 2*self.svm_kernel(self.kernel,dataMatrix[i],dataMatrix[j])- \
					self.svm_kernel(self.kernel,dataMatrix[i],dataMatrix[i])- \
					self.svm_kernel(self.kernel,dataMatrix[j],dataMatrix[j])
					if eta >= 0:
						continue
					alphas[j] = alphas[j] - labelMatrix[j]*(E[i]-E[j])/eta
					if alphas[j] > H:
						alphas[j] = H
					elif alphas[j] < L:
						alphas[j] = L
					if abs(alphaJold-alphas[j])<0.00001:
						continue
					alphas[i] = alphas[i] + labelMatrix[i]*labelMatrix[j](alphaJold-alphas[j])
					b1 = b - E[i] - labelMatrix[i]*(alphas[i] - alphaIold)*self.svm_kernel(self.kernel,dataMatrix[i],dataMatrix[i]) \
					labelMatrix[j]*(alphas[j] - alphaJold)*self.svm_kernel(self.kernel,dataMatrix[i],dataMatrix[j])
					b2 = b - E[j] - labelMatrix[i]*(alphas[i] - alphaIold)*self.svm_kernel(self.kernel,dataMatrix[i],dataMatrix[j]) \
					labelMatrix[j]*(alphas[j] - alphaJold)*self.svm_kernel(self.kernel,dataMatrix[j],dataMatrix[j])
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
		#计算W参数,W为向量
		w = 0
		for i in range (m):
			w += alphas[i]*dataMatrix[i]*labelMatrix[i]
		return w,alphas,b







		
