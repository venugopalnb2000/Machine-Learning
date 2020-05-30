import random as rd
import math as m
# import pycode_similar
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import time
ratio=0.8
# ratio=float(input("Enter ratio "))/100.0
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

def load_split(file):
	line=open(file,'r').read().strip().splitlines()
	l=[]
	for i in line[1:]:
		a=[]
		for j in i.split(","):
			a.append(float(j))
		l.append(a)
	# k=[]
	# k=[i[:14] for i in l]
	# print(k)
	# print(l)
	training_size=len(l)*ratio
	training_set=[]
	# print(l)
	for _ in range(int(training_size)):
		training_set.append(l.pop(rd.randrange(len(l))))
	return [training_set,l]
def get_mean(feature):
	return sum(feature)/float(len(feature))
def get_std_dev(feature):
	mean=get_mean(feature)
	std_dev=m.sqrt(sum([pow(x-mean,2) for x in feature])/float(len(feature)-1))
	return std_dev
def summarize_data(data):
	class_0_1={}
	for i in data:
		if(i[-1] not in class_0_1):
			class_0_1[i[-1]]=[]
		k=i[-1]
		del i[-1]
		class_0_1[k].append(i)
	pv={}
	for c,i in class_0_1.items():
		pv[c]=[[get_mean(attr),get_std_dev(attr)] for attr in zip(*i)]
	return pv
def prob(x,mean,stdev):
	if stdev==0:
		return 1
	ex=m.exp(-(m.pow(x-mean,2)/(2*m.pow(stdev ,2))))
	a=(1/(m.sqrt(2*m.pi)*stdev))*ex
	return a
def predict(pv,test1):
	pb={}
	best_label,best_prob=None,-1
	for cv,cs in pv.items():
		pb[cv]=1
		for i in range(len(cs)):
			mean,stdv=cs[i]
			pb[cv]*=prob(test1[i],mean,stdv)
		# pb[cv]=[pb[cv]*prob(x,mean,stdv) for mean,stdv in cs]
	for cv1,pb1 in pb.items():
		if(best_label is None or pb1>best_prob):
			best_prob=pb1
			best_label=cv1
	# print(best_label)
	return best_label
def get_pred(pv,testing_set):
	predicted=[]
	y_true=[]
	predicted=[predict(pv,testing_set[x]) for x in range(len(testing_set))]
	y_true=[i[-1] for i in testing_set]	
	return [predicted,y_true]
def get_accuracy(pv,testing_set):
	c=0
	predicted,y_true=get_pred(pv,testing_set)
	auc_roc_curve(predicted,y_true)
	# print(predicted)
	for i in range(len(y_true)):
		if(predicted[i]==y_true[i]):
			c+=1
	return (c/float(len(testing_set)))*100
def main():
	import sys
	if(len(sys.argv)<2):
		print("File name missing")
		sys.exit(0)
	else:
		file=sys.argv[1]
	print(file)
	training_set,testing_set=load_split(file)
	pv=summarize_data(training_set)
	accuracy=get_accuracy(pv,testing_set)
	print("Accuracy",accuracy)
	
main()