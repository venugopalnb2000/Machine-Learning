import random as rd
from sklearn . metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn . metrics import classification_report
import copy
import time
import math as m


ratio=0.80

def load(file):
	l=open(file,'r').read().strip().splitlines()
	data=[]
	for i in l[1:]:
		a=i.split(",")
		data.append([float(j) for j in a])
	return data

def load_split(file):
	data=load(file)
	train=[]
	total_data=len(data)
	while(len(train)<int(ratio*total_data)):
		train.append(data.pop(rd.randrange(len(data))))
	return [train,data]

def euclidean_distance(a,b):
	dist=[pow((a[i]-b[i]),2) for i in range(len(a)-1)]
	return [m.sqrt(sum(dist)),b[-1]]

def k_min_distance(k,test_1,train_all):
	a=[euclidean_distance(test_1,train_all[i]) for i in range(len(train_all))]
	a.sort(key=lambda x: x[0])
	return a[:k]

def predict_class(a):
	class_0=0
	class_1=0
	for i in a:
		if(i[1]==0.0):
			class_0+=1
		else:
			class_1+=1
	if(class_0>class_1):
		return 0.0
	return 1.0

def predict_test(k,traino,testo):
	predict=[]
	for i in range(len(testo)):
		a=k_min_distance(k,testo[i],traino)
		predict.append(predict_class(a))
	return predict

def get_num(train,num):
	c=0
	for i in range(len(train)):
		if(train[i][-1]==num):
			c+=1
	return c

def accuracy_predict(tv,predo):
	c=0
	for i in range(len(predo)):
		if(tv[i]==predo[i]):
			c+=1
	return (c/len(tv))*100

def upsample_train(train):
	ones=get_num(train,1.0)
	h=get_num(train,0.0)
	while(h<ones):
		k=rd.randrange(len(train))
		if(train[k][-1]==0.0):
			train.append(train[k])
			h+=1
	return train

def cross_validate(km,data):
	va=10
	akka=[]
	data_split=[]
	fs=int(len(data)/va)
	for i in range(va-1):
		f=[]
		while(len(f)<fs):
			f.append(data.pop(rd.randrange(len(data))))
		data_split.append(f)

	data_split.append([i for i in data])
	d_copy=copy.deepcopy(data_split)

	for i in range(len(data_split)):
		test=d_copy[i]
		del d_copy[i]
		train=[]
		for k in d_copy:
			for j in range(fs):
				train.append(k[j])
		true=[ip[-1] for ip in test]
		predict=predict_test(km,train,test)
		akka.append(accuracy_predict(predict,true))
		d_copy=copy.deepcopy(data_split)

	return sum(akka)/len(akka)

def confusion_matrix(pred,tv):
	tp=0
	fp=0
	fn=0
	tn=0
	l=[]
	for i in range(len(pred)):
		if(tv[i]==1 and pred[i]==1):
			tp+=1
		elif(tv[i]==1 and pred[i]==0):
			tn+=1
		elif(tv[i]==0 and pred[i]==1):
			fp+=1
		else:
			fn+=1
	total_1=tv.count(1.0)
	total_0=tv.count(0.0)
	l=[[tp,tn],[fp,fn]]
	print("Confusion matrix ")
	for i in l:
		print(str(i[0])+"   "+str(i[1]))

	accuracy=((tp+fn)/(total_0+total_1))*100
	precision=tp/(tp+fp)
	recall=tp/(tp+fn)
	f1_score=(2*precision*recall)/(precision+recall)

	print("\nOverAll\nAccuracy : %0.2f"%accuracy,"\nPrecision : %0.2f"%precision,"\nRecall : %0.2f"%recall,"\nF1 Score : %0.2f"%f1_score)

	print("\nClasswise Accuracy")

	print("class_0 : %0.2f"%((fn/total_0)*100))
	print("class_1 : %0.2f"%((tp/total_1)*100))


def auc_roc_curve(predict,true):

	fpr,tpr,threshold=roc_curve(predict,true)
	roc_auc=auc(fpr,tpr)

	plt.figure()
	plt.plot(fpr,tpr,color='red',label='Roc curve (area= %0.2f)'%roc_auc)
	plt.plot([0,1],[0,1],linestyle='--')
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc='lower right')
	plt.show()


def main():
	
	k=3
	# file="cat2_pakka.csv"
	import sys
	if(len(sys.argv)<2):
		print("File name missing")
		sys.exit(0)
	else:
		file=sys.argv[1]
	print(file)
	train,test=load_split(file)
	true=[i[-1] for i in test]

	train=upsample_train(train)
	starttime=time.time()
	predict=predict_test(k,train,test)

	confusion_matrix(predict,true)	

	endtime=time.time()-starttime

	print("\nTime(s) for calulating Accuracy : %0.4f"%endtime)
	print("\n")

	print("K-fold Cross validation : %0.2f"%cross_validate(k,load(file)))
	auc_roc_curve(predict,true)


main()
