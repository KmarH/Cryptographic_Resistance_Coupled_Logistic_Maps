# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn import svm
import matplotlib.pyplot as plt
import sys
import binascii
import xlsxwriter
import time
from matplotlib.backends.backend_pdf import PdfPages
#import threading
#import time
#import datetime
#import os
#import random
#import Tkinter as Tk
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from hmmlearn.hmm import GaussianHMM, MultinomialHMM


#XExcel File
workbook = xlsxwriter.Workbook('exp.xlsx')
worksheet = workbook.add_worksheet()
pp = PdfPages('m1.pdf')

m1=bin(int(binascii.hexlify(bytes(sys.argv[1],'utf-8')), 16))
#print(m1)

#Logistic Map
def log_map(v, mu):
	return mu * v * (1-v)

#start Simulation
start_time_sim = time.time()

#Encryption Function
x = []
y = []

#Parameters
p = 50
stpoint = 4000
flpoint = 150500
#Keys 2.9 / .3314
#III mu = 3.54 mu1 = 2.9 alpha = 0.1214
mu = 3.1
mu1 = 2.9
alpha = 0.3314

#x0= np.random.rand()
#y0= np.random.rand()

x0= 0.03
y0= 0.01

x.append(x0)
y.append(y0)

#Coupled Map Simulation
for i in range(1,flpoint):
	xn = log_map(x[i-1],mu) + (alpha *(y[i-1]-x[i-1]))
	yn = log_map(y[i-1],mu1) + (alpha *(x[i-1]-y[i-1]))
	x.append(xn)
	y.append(yn)
	worksheet.write(i, 0, xn)
	worksheet.write(i, 1, yn)

#Simulated table
#XY = np.array(zip(x,y))
XY=np.column_stack((x,y))
#print(XY)
#XXY=np.array(sorted(XY, key=lambda lu: lu[0]))
XXY=XY[XY[:, 0].argsort()]

#plt.subplot(2, 1, 1)
plt.figure(figsize=(6, 6))
plt.scatter(x, y, 0.0001)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(pp, format='pdf')
plt.savefig('fig3.png', format='png', dpi=300)
#print("--- %s seconds ---" % (time.time() - start_time_sim))
#plt.show()
X = np.array([[0,0],[0,0]])
def encryp(m,pp):
	#plt.legend(loc=2)
	pp1 = int(stpoint + (int(pp) * 50))
	pp2 = int(pp1+50)

	#To the transmitter
	y_r = y[pp1:pp2]
	#y_r = np.random.rand(10000)
	x_t =[]
	x_t.append(x[(pp1)])
	#x_t.append(x[10349])
	#x_t.append(np.random.rand()/5)
	#Message encryption
	#print(m)
	worksheet.write(0, 2+(2*int(pp)), m)
	for i in range(1,50):
		xn = log_map(x_t[i-1],mu) + (alpha *(y_r[i-1]-x_t[i-1])) + (0.01*m)
		x_t.append(xn)
		worksheet.write(i, 2+(2*int(pp)), xn)
		worksheet.write(i, 3+(2*int(pp)), y_r[i-1])
	if m==1:
		plt.scatter(x_t, y_r, color = 'darkblue', s = 10)
	else:
		plt.scatter(x_t, y_r, color = 'red', s = 10)
	#Z = np.array(zip(x_t[1:], y[pp1:pp2-1])) // , color = 'darkblue', marker='x'
	#Z = np.array(zip(x_t, y_r))
	Z = np.column_stack((x_t,y_r))
	global X
	X = np.concatenate((X,Z))
	return x_t

def unencryp(x_t,pp):
	pp1 = int(stpoint + (int(pp) * 50))
	pp2 = int(pp1+50)
	y_r = y[pp1:pp2]
	jj = 1
	jjj = []
	for i in range(0,49):
		#pXXY = int((x_t[i]-XXY_min)*DXXY)
		pXXY = (np.abs(XXY[:,0]-x_t[i])).argmin()
		if pXXY < 150:
			pXXY = 0
		if pXXY > (flpoint-151):
			pXXY = flpoint-151
		for j in range(pXXY-150, pXXY+150):
			myList = (((x_t[i] - XXY[j,0])**2 + (y_r[i] - XXY[j,1])**2)**0.5)
			if j==(pXXY-150):
				jj = myList
			if myList < jj:
				jj = myList
			#jj=min(myList)
		jjj.append(jj)
		#print(x_t[i])
		#print(jj)
	if np.mean(jjj[37:49]) > 0.000913333:
		return(1)
	else:
		return(0)

nn = ''
m2 = m1[2:]
cont_1 = 0

Y = np.array([0,0])
plt.figure(figsize=(6, 6))
plt.scatter(x, y, 0.0001)

#print(X)
#start Encrypt
start_time_en = time.time()
s_t = []
for i in m2:
	s_t1 = encryp(int(i),cont_1)
	pp1 = int(stpoint + (int(i) * 50))
	pp2 = int(pp1+50)
	cont_1 = cont_1 + 1
	s_t.append(s_t1)
    #Z = np.array(zip(s_t,y[(pp1-51):(pp2-51)]))
    #X = np.concatenate((X,Z))
#print("--- %s seconds ---" % (time.time() - start_time_en))

cont_1 = 0
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(['Simulation','1','0'])
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig(pp, format='pdf')
plt.savefig('fig1.png', format='png', dpi=300)
#plt.show()

plt.figure(figsize=(6, 6))
plt.xlim(0, 50)
plt.ylim(0, 1)
plt.xlabel('n');
plt.ylabel('Signal of the transmitter');
plt.plot(s_t[0],'bo')
plt.plot(s_t[1],'ro')
plt.legend(['1','0'])
plt.savefig(pp, format='pdf')
plt.savefig('fig2.png', format='png', dpi=300)
#plt.show()
#start Decrypt
start_time_den = time.time()
for i in m2:
    nnn = unencryp(s_t[cont_1],cont_1)
    if nnn == 0:
        Y=np.append(Y, np.zeros((50,), dtype=int),axis=0)
    else:
        Y=np.append(Y, np.ones((50,), dtype=int),axis=0)
    nn=nn+str(nnn)
    cont_1 = cont_1 + 1
#print(nn)
#print("--- %s seconds ---" % (time.time() - start_time_den))
#print(nn)
nn = '0b' + nn
n = int(nn, 2)

#print(n)
de = binascii.unhexlify('%x' % n)
#print(nn)
#print(de)

#plt.subplot(2, 1, 2)



#load the dataset
#data = loadtxt('ex2data2.txt', delimiter=',')

#X = data[:, 0:2]
#y = data[:, 2]

def render_tests(data, accepted, rejected):
	plt.figure(figsize=(6, 6))
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.scatter(np.take(data[:,0], accepted),
            np.take(data[:,1], accepted),
            c='b', marker='x', label='1', s=25)
	plt.scatter(np.take(data[:,0], rejected),
            np.take(data[:,1], rejected),
            c='y', marker='o', label='0', s=5)
	plt.xlabel('X');
	plt.ylabel('Y');
	plt.axes().set_aspect('equal', 'datalim')
	#plt.savefig(pp, format='pdf')


X = X[2:]
Y = Y[2:]

accepted = np.where(Y == 1)
rejected = np.where(Y == 0)

render_tests(X, accepted, rejected)


workbook.close()




