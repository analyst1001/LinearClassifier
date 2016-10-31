#!/bin/python
import random
import numpy as np
class DataItem:
	def __init__(self):
		self.classLabel = 0
		self.feature = []


class Dataset:
	def __init__(self):
		self.data = []
	def readData(self,filename):
		f = open(filename)
		label = {}
		for line in f:
			line = line.split(",")
			if line[-1] not in label:
				label[line[-1]] = len(label)
			item = DataItem()
			item.classLabel = label[line[-1]]
			for i in range(len(line) - 1):
				item.feature.append(float(line[i]))
			item.feature = [1] + item.feature
			self.data.append(item)
		f.close()


def splitDataset(complete,folds,seeds):
	split_data = []
	for i in range(folds):
		item = Dataset()
		split_data.append(item)
	random.seed(seeds)
	act_len = len(complete.data)
	for i in range(folds):
		for j in range(act_len/folds):
			k = random.randint(0,len(complete.data)-1)
			split_data[i].data.append(complete.data[k])
			del(complete.data[k])
	return split_data


def mergeDatasets(toMerge,numDatasets,indicesToMerge):
	merge = Dataset()
	for i in range(numDatasets):
		for j in range(len(toMerge[i].data)):
			merge.data.append(toMerge[i].data[j])
	return merge


class LinearClassifier:

	def __init__(self,):
		self.numClasses = 0
		self.a = []
		self.training_error = 0
		self.comb = 0
		self.algo = 0
		self.feature_num = 0
		self.margin = 0
		self.cycles = 0

	def batch_perceptron_variable_eta(self,complete):
		b = [1.0] * self.feature_num
		error = 0.05
		k = 0
		while (1):
			k = k + 1
			sum_y = [0] * self.feature_num
			no_mis = 0
			for m in range(len(complete.data)):
				if (np.dot(complete.data[m].feature,b)<0):
					sum_y = [i+j for i,j in zip(sum_y,complete.data[m].feature)]
					no_mis = no_mis + 1
			b = [i+j for i,j in zip(b,map(lambda x: (1.0/k)*x, sum_y))]
			if ((no_mis*1.0)/len(complete.data)<error):
				error = (no_mis*1.0)/(len(complete.data))
				break
			if (k>self.cycles):
				error = (no_mis*1.0)/len(complete.data)
				break
		return (error,b)

	def single_sample_fixed_perceptron(self,complete):
		b = [1.0] * self.feature_num
		error = 0.1
		c = 0
		k = 0
		while (1):
			k = k%len(complete.data)
			if (np.dot(b,complete.data[k].feature)<0):
				b = [i+j for i,j in zip(b,complete.data[k].feature)]
				no_mis = 0
				for i in range(len(complete.data)):
					if (np.dot(b,complete.data[i].feature)<0): no_mis = no_mis + 1
				if ((no_mis*1.0)/len(complete.data)<error):
					error = (no_mis*1.0)/len(complete.data)
					break
			k = k + 1
			c = c + 1
			if (c>self.cycles):
				no_mis = 0
				for i in range(len(complete.data)):
					if (np.dot(b,complete.data[i].feature)<0): no_mis = no_mis + 1
				error = (no_mis*1.0)/len(complete.data)
				break
		return (error,b)
				
	def single_sample_variable_relaxation(self,complete):
		b = [1.0] * self.feature_num
		error = 0.1
		c = 0
		while (1):
			misclassified = filter(lambda x: sum(np.array(x)*np.array(b)) <= self.margin, map(lambda x: x.feature,complete.data))
			if (len(misclassified)==0):
				error = 0
				break
			elif (len(misclassified)*1.0/len(complete.data)<error):
				error = len(misclassified)*1.0/len(complete.data)
				break
			y = misclassified[0]
			c = c + 1
			value = ((1.0/c)*(self.margin - sum(np.array(b)*np.array(y))))/sum(np.array(y)*np.array(y))
			b = list(np.array(b) + (value*np.array(y)))
			if (c>self.cycles):
				error = (len(misclassified)*1.0)/len(complete.data)
				break
		return (error,b)
	
	def batch_variable_relaxation(self,complete):
		b = [1.0] * self.feature_num
		error = 0.1
		k = 1
		while (1):
			no_mis = 0
			for m in range(len(complete.data)):
				value = np.dot(b,complete.data[m].feature)
				if (value<self.margin):
					y = complete.data[m].feature[:]
					square_y = sum([i*i for i in y])
					inc = (1.0/(k))*(1.0*(self.margin - value)/square_y)
					b = [i+j for i,j in zip(b,map(lambda x: inc * x,y))]
					no_mis = no_mis + 1
			if ((no_mis*1.0)/len(complete.data)<error):
				error = (no_mis*1.0)/len(complete.data)
				break
			k = k + 1
			if (k>self.cycles):
				error = (no_mis*1.0)/len(complete.data)
				break
		return (error,b)

	def onevsrest(self,complete):
		for i in range(self.numClasses):
			argument = Dataset()
			for j in range(len(complete.data)):
				new_dataset = DataItem()
				new_dataset.classLabel = complete.data[j].classLabel
				new_dataset.feature = complete.data[j].feature[:]
				argument.data.append(new_dataset)
				if (complete.data[j].classLabel != i):
					argument.data[j].feature = map(lambda x: (-1)*x,argument.data[j].feature)
					argument.data[j].classLabel = -1
			if (self.algo == 1):
				(error,b) = self.single_sample_fixed_perceptron(argument)
				self.a.append(b)
				self.training_error = error + self.training_error
			elif (self.algo == 2):
				(error,b) = self.batch_perceptron_variable_eta(argument)
				self.a.append(b)
				self.training_error = error + self.training_error
			elif (self.algo == 3):
				(error,b) = self.single_sample_variable_relaxation(argument)
				self.a.append(b)
				self.training_error = error + self.training_error
		self.training_error = (self.training_error*1.0)/self.numClasses

	def onevsone(self,complete):
		self.a = {}
		for i in range(self.numClasses):
			for j in range(i+1,self.numClasses):
				argument = Dataset()
				for k in range(len(complete.data)):
					new_sample = DataItem()
					if (complete.data[k].classLabel == i):
						new_sample.classLabel = i
						new_sample.feature = complete.data[k].feature[:]
						argument.data.append(new_sample)
					elif (complete.data[k].classLabel == j):
						new_sample.classLabel = j
						new_sample.feature = complete.data[k].feature[:]
						argument.data.append(new_sample)
						argument.data[-1].feature = map(lambda x: (-1)*x,argument.data[-1].feature)
				if (self.algo == 1):
					(error,b) = self.single_sample_fixed_perceptron(argument)
					if ((i,j) not in self.a):
						self.a[(i,j)] = []
					self.a[(i,j)] = b[:]
					self.training_error = error + self.training_error
				elif (self.algo == 2):
					(error,b) = self.batch_perceptron_variable_eta(argument)
					if ((i,j) not in self.a):
						self.a[(i,j)] = []
					self.a[(i,j)] = b[:]
					self.training_error = error + self.training_error
				elif (self.algo == 3):
					(error,b) = self.single_sample_variable_relaxation(argument)
					if ((i,j) not in self.a):
						self.a[(i,j)] = []
					self.a[(i,j)] = b[:]
					self.training_error = error + self.training_error
		self.training_error = (self.training_error*2.0)/((self.numClasses)*(self.numClasses-1))
	
	def ddag(self,complete):
		self.onevsone(complete)	
	
	def ddag_classify(self,start,end,sample):
		if (start == end): return start
		val = np.dot(sample.feature,self.a[(start,end)])
		if (val >0):
			return self.ddag_classify(start,end-1,sample)
		else:
			return self.ddag_classify(start+1,end,sample)
	
	def classifySample(self,sample):
		if (self.comb == 1):
			answer = []
			for i in range(self.numClasses):
				if (np.dot(sample.feature,self.a[i])>0):answer.append(i)
			if (len(answer)>0): return random.choice(answer)
			else: return random.randint(0,self.numClasses-1)
		elif (self.comb == 2):
			answer = []
			for i in self.a:
				if (np.dot(sample.feature,self.a[i])>0):answer.append(i[0])
				if (np.dot(sample.feature,self.a[i])<0):answer.append(i[1])
			return max(answer, key = answer.count)
		elif (self.comb == 3):
			return (self.ddag_classify(0,self.numClasses-1,sample))

	def classifyDataset(self,testSet, cm):
		error = 0
		for i in range(len(testSet.data)):
			derived = self.classifySample(testSet.data[i])
			actual = testSet.data[i].classLabel
			cm[derived][actual] = cm[derived][actual] + 1
			if (actual != derived):
				error = error + 1
		return (error*1.0)/len(testSet.data)

	def learnModel(self,complete,algorithm,combination):
		self.algo = algorithm
		self.comb = combination
		if (combination == 1):
			self.onevsrest(complete)
		elif (combination == 2):
			self.onevsone(complete)
		elif (combination == 3):
			self.ddag(complete)
		return self.training_error
	
	def saveModel(self,filename):
		f = open(filename,'w')
		pickle.dump(self.a,f)
		f.close()
	
	def LoadModel(self,filename):
		f = open(filename,'r')
		for line in f:
			if "margin" in line:
				self.margin = float(line.split(" ")[2])
			elif "cycles" in line:
				self.cycles = int(line.split(" ")[2])
			elif "number_of_classes" in line:
				self.numClasses = int(line.split(" ")[2])
			elif "number_of_features" in line:
				self.feature_num = int(line.split(" ")[2]) + 1
		f.close()


def crossValidate(complete,folds,algo,comb):
	split = splitDataset(complete,folds,algo)
	errors = []
	training_error = []
	avg_cm = []
	for i in range(folds):
		l = range(folds)
		del(l[i])
		merge = mergeDatasets(split,folds,l)
		x = LinearClassifier()
		x.LoadModel("load")
		t = x.learnModel(merge,algo,comb)
		training_error.append(t)
		cm = np.array([[0]*x.numClasses]*x.numClasses)
		if (len(avg_cm) == 0):
			avg_cm = np.array([[0]*x.numClasses]*x.numClasses)
		errors.append(x.classifyDataset(split[i],cm))
		avg_cm = avg_cm + cm
	avg_cm = avg_cm*(1.0/folds)
	return (np.average(errors),np.std(errors),avg_cm,np.average(training_error))


def main():
	filename = "iris.txt"
	ds = Dataset()
	ds.readData(filename)
	print "Enter algo"
	algo = input()
	print "Enter combination"
	comb = input()
	(test_avg,test_deviation,cm,training_error) = crossValidate(ds,5,algo,comb)
	print "The average training error is"
	print training_error
	print
	print "The average test error is"
	print test_avg
	print
	print "The test errors standard deviation is"
	print test_deviation
	print
	print "The average confusion matrix for test samples is"
	print cm
	print


if __name__ == "__main__":
	main()
