import numpy as np

class tree_node():

	def __init__(self,):
		self.left = None
		self.right = None
		self.val = None
		self.col = None



class decesion_tree():

	def __init__(self,depth,number):
		self.depth = depth
		self.leaf_number = number

	def divide_set(data,column,value,n):
		splittingFunction = None
		set1 = np.empty((0,n))
		set2 = np.empty((0,n))
		if isinstance(value, int) or isinstance(value, float): # for int and float values
			splittingFunction = lambda row : row[column] >= value
		else: # for strings 
			splittingFunction = lambda row : row[column] == value
		for row in data:
			if splittingFunction(row):
				set1 = np.concatenate((set1, [row]), axis=0)
			else:
				set2 = np.concatenate((set2, [row]), axis=0)
		return set1,set2

	def gini(data):
		#如果知道了label的种类的话可以直接使用label，不需要经过下面计算
		label = np.unique(data[:,[-1]])
		n = len(data)
		label_count = [0]
		p = 0
		for row in data:
			label_count[row[-1]]++
		for data in label_count:
			p += (float(data)/n)**2
		return 1 - p

	def create_tree(data,depth,select_feature):
		'''结束条件：
		1.树的深度达到用户指定的要求
		2.叶子节点的样本数少于指定的阈值
		3.信息增益，信息增益率，gini指数的变化小于一定的阈值
		'''
		#特别注意data, class_label必须是numpy array
		
		m,n = np.shape(data)
		best_gini = 10000
		best_attribute = None
		best_sets = None
		#节点样本数目小于20停止划分
		if m < = 20:
			return tree_node()
		#树的深度达到一定要求，停止划分
		if depth <= 1 :
			return tree_node()
		#获取某一列的特征,可以通过并行计算对下面计算进行优化,最后一项是label
		origin_gini = self.gini(data)
		#需要减去最后一列，因为最后一列是label
		for i in range(n-1):
			if i not in select_feature:
				feature = data[:,[i]]
			#选择所有可能的分割点(遍历每一个feature所有可能的取值)
				for value in np.unique(feature):
				#通过value来将数据集划分为两部分
					set1,set2 = self.divide_set(data,i,value,n)
				#计算其gini指数，如果大于默认值则保存下来
					p = len(set1)/m
					gini = p*self.gini(set1)+(1-p)*self.gini(set2)
					if gini < best_gini :
						best_gini = gini
						best_sets = (set1,set2)
						best_attribute = (i,value)
		if abs(origin_gini,best_gini) > tol:
			select_feature.append(best_attribute[0])
			right = create_tree(set1,depth-1,select_feature)
			left = create_tree(set2,depth-1,select_feature)
			return tree_node(col=best_attribute[0], value=best_attribute[1], right=right, left=left)
		else:
			return tree_node()





			#计算gini指数


	def fit(self ,data , class_label, sample_weight=None):
		#data and class_label are numpy array
		data = np.concatenate((data, class_label), axis=1)
		#select_feature表示已经作为分类的特征不能用于其他层的分类(这种做法不妥，比如特征Ａ,第一层的分类安装Ａ<10分为两类，第二层可以继续Ａ>3继续进行划分)
		#所以不应该使用select_feature
		self.create_tree(data,self.depth,select_feature=[])
