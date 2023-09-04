import numpy as np
import pandas as pd
import numpy.linalg as la
import copy

def circle_3(img,x0,y0,z0,r):

	temp = copy.deepcopy(img) 
	m,n,d = img.shape
	for i in range(m):
		for j in range(n):
			for k in range(d):
				dist = np.round(la.norm(np.array([i-x0, j-y0, k - z0]),2))
				if dist <= r:
					temp[i,j,k] = 1

	return temp

def rectangle(img,x0,y0,r):
	m,n = img.shape
	img[int(x0-r):int(x0),int(y0):int(y0+r)] = 1
	return img

def circle(img,x0,y0,r):
	temp = copy.deepcopy(img)
	m,n = temp.shape
	for i in range(m):
		for j in range(n):
			dist = np.round(la.norm(np.array([i-x0, j-y0]),2))
			if dist <= r:
				temp[i,j] = 1
	return temp
## generate concentric circle
def con_circle(img,x0,y0,r1,r2):
	temp = copy.deepcopy(img)
	m,n = temp.shape
	for i in range(m):
		for j in range(n):
			dist = np.round(la.norm(np.array([i-x0, j-y0]),2))
			if dist >= r1 and dist <= r2:
				temp[i,j] = 1

	return temp
def _T(img,x0,y0,length,width):
	m,n = img.shape
	r1 = int(length/2)
	r2 = int(width/2)
	img[int(x0):int(x0+length*0.8),int(y0-r2):int(y0+r2)] = 1
	img[int(x0-width):int(x0),int(y0-r1):int(y0+r1)] = 1                
	return img

def _cross(img,x0,y0,length,width):
	m,n = img.shape
	r1 = int(length/2)
	r2 = int(width/2)
	img[int(x0-r1-r2):int(x0+r1-r2),int(y0+r1-r2):int(y0+r1+r2)] = 1
	img[int(x0-width):int(x0),int(y0):int(y0+length)] = 1 
	return img


def triangle(img,x0,y0,r):
	m,n = img.shape
	x0 = x0
	y0 = y0
	h = np.round(np.sqrt(3)/2 * r)
	x1,y1 = x0 - r/2, y0
	x2,y2 = x0 + r/2, y0
	e1 = (np.array([x0-x1,y0-y1])/la.norm(np.array([x0-x1,y0-y1]),2)).reshape(-1,1)
	e2 = (np.array([x0-x2,y0-y2])/la.norm(np.array([x0-x2,y0-y2]),2)).reshape(-1,1)
	
	for i in range(m):
		for j in range(n):
			if i <= y0:
				vec1 = np.array([i-x1,j-y1]).reshape(-1,1)
				dist1 = np.round(la.norm(vec1,2))
				_norm1 = vec1/dist1
				vec2 = np.array([i-x2,j-y2]).reshape(-1,1)
				dist2 = la.norm(vec2,2)
				_norm2 = np.round(vec2/dist2)
				_theta1 = np.trace(_norm1.T.dot(e1))
				_theta2 = np.trace(_norm2.T.dot(e2))
				if _theta1 >= np.sqrt(2)/2 and _theta2 >= np.sqrt(2)/2 and (np.abs(i-x1) != np.abs(j-y0)):
					img[i,j] = 1
	return img