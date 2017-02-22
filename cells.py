import cv2
import numpy as np

cap = cv2.VideoCapture('cells.avi')

while(cap.isOpened()):
	ret,frame = cap.read()
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	lower_purple = np.array([130,50,140])
	upper_purple = np.array([170,255,255])
	mask = cv2.inRange(hsv, lower_purple, upper_purple)
	res = cv2.bitwise_and(frame,frame, mask= mask)
	blur = cv2.GaussianBlur(res,(7,7),0)
	kernel = np.ones((5,5),np.uint8)
	dilation = cv2.dilate(blur,kernel,iterations = 1)
	edges = cv2.Canny(dilation,100,200)
	contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		epsilon = 0.1*cv2.arcLength(cnt,True)		# Contour approximation to get lesser corners
		approx = cv2.approxPolyDP(cnt,epsilon,True)	# array containing the approximated contours
		cv2.drawContours(edges,approx,-1,(255,128,0),5)
	cv2.imshow('edges',edges)
	#cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()	
