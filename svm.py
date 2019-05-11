#编写svm源码解决实际问题（先自己想，不懂再参看别人的代码）
#解决什么实际问题（元动作识别，ride,run,walk,sit）
#关键点是smo算法
#第一步是数据的读取和存储，然后初始化参数，为需要的变量分配空间，然后就是伪代码的实现

# read data and store data
# 1)样本数量不大，万级别　　2)巨大数量的样本，十万百万级别
# 样本数量不大，直接用pandas读取数据文件进入内存，然后对数据进行预处理，将数据划分为测试集和训练集



# 为svm模型建立一个类


class SVM (object):
	"""SVM中常见参数Ｃ，gramma,核函数类型，tol,最大迭代次数max_iter

SVM """
	def __init__(self, C,gramma,tol,max_iter):

		self.arg = arg

	def calculate_error(self,index):


	def fit(self ,X, y, sample_weight=None):
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

		#初始化参数a,b ,pandas用于统计分析(类似于表格),numpy常用于数值计算，此处我们假设X为numpy矩阵
		length = X.shape[1]
		a = [0 for i in range(length)]
		#a = np.zeros(length)
		b = 0
		passes = 0
		Error = [0 for i in range(length)]
		while passes < max_passes:
			num_changed_alphas = 0
			# row 是 X　二维矩阵中的每一行
			for index in range(length):
				Error[index] = self.calculate_error(index)







		
