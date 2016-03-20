import cv2
import Image
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from SocketServer import ThreadingMixIn
import StringIO
import time
import math
import numpy as np
from common import clock, draw_str
import sys


from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils

vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

class CamHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		print self.path
		if (self.path == "/cam.mjpg"):
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			t = clock()
			while fps:
			
				img = vs.read()
				dt = clock() - t
				t = clock()
				draw_str(img, (20, 20), 'time: %.1f ms' % (dt*1000))

				imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
				jpg = Image.fromarray(imgRGB)
				tmpFile = StringIO.StringIO()
				jpg.save(tmpFile,'JPEG')
				self.wfile.write("--jpgboundary")
				self.send_header('Content-type','image/jpeg')
				self.send_header('Content-length',str(tmpFile.len))
				self.end_headers()
				jpg.save(self.wfile,'JPEG')
				#time.sleep(0.05)
				
			return

		elif (self.path == "/finger_count/cam.mjpg"):
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			t = clock()
			while fps:

				img = vs.read()

				cv2.rectangle(img,(300,300),(100,100),(0,255,0),0)
				crop_img = img[100:300, 100:300]  
				grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  
				value = (35, 35)  
				blurred = cv2.GaussianBlur(grey, value, 0)  
				ret, thresh1 = cv2.threshold(blurred, 0, 255,
			                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  
				#cv2.imshow('Thresholded', thresh1)  
				contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
			            cv2.CHAIN_APPROX_NONE)  
				drawing = np.zeros(crop_img.shape,np.uint8)  

				max_area = 0
			    
				for i in range(len(contours)):  
					cnt=contours[i]
			        area = cv2.contourArea(cnt)
			        if(area>max_area):
			            max_area=area
			            ci=i  
				cnt=contours[ci]  
				x,y,w,h = cv2.boundingRect(cnt) 
				cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0) 
				hull = cv2.convexHull(cnt) 
				cv2.drawContours(drawing,[cnt],0,(0,255,0),0) 
				cv2.drawContours(drawing,[hull],0,(0,0,255),0)  
				hull = cv2.convexHull(cnt,returnPoints = False)  
				defects = cv2.convexityDefects(cnt,hull)  
				count_defects = 0  
				cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)  
				for i in range(defects.shape[0]):  
					s,e,f,d = defects[i,0]
			        start = tuple(cnt[s][0])
			        end = tuple(cnt[e][0])
			        far = tuple(cnt[f][0])
			        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
			        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
			        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
			        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
			        if angle <= 90:
			            count_defects += 1
			            cv2.circle(crop_img,far,1,[0,0,255],-1)
			        #dist = cv2.pointPolygonTest(cnt,far,True)
			        cv2.line(crop_img,start,end,[0,255,0],2)
			        #cv2.circle(crop_img,far,5,[0,0,255],-1)    
				if count_defects == 0:  
					cv2.putText(img,"1 Finger", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)  
				if count_defects == 1:  
					cv2.putText(img,"2 Fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)  
				elif count_defects == 2:  
					cv2.putText(img, "3 fingers", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)    
				elif count_defects == 3:  
					cv2.putText(img,"4 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)  
				elif count_defects == 4:  
					cv2.putText(img,"5 fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)  
				elif count_defects == 5:  
					cv2.putText(img,"impossible", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

				dt = clock() - t
				t = clock()
				draw_str(img, (20, 20), 'time: %.1f ms' % (dt*1000))

				imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
				jpg = Image.fromarray(imgRGB)
				tmpFile = StringIO.StringIO()
				jpg.save(tmpFile,'JPEG')
				self.wfile.write("--jpgboundary")
				self.send_header('Content-type','image/jpeg')
				self.send_header('Content-length',str(tmpFile.len))
				self.end_headers()
				jpg.save(self.wfile,'JPEG')
				#time.sleep(0.05)

			return


		elif (self.path == "/track_object/cam.mjpg"):
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			t = clock()

			img = image
			r, h, c, w = 190, 90, 300, 120
			track_window = (c,r,w,h)

			roi = img[r:r+h, c:c+w]
			hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
			roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
			cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
			term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

			while (1):
				
				
				img = image
				hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  
				dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)  
				ret, track_window = cv2.CamShift(dst, track_window, term_crit)
				
				x,y,w,h = track_window
				img2 = cv2.rectangle(img, (x,y), (x+w,y+h), 255,2)

				dt = clock() - t
				t = clock()
				draw_str(img, (20, 20), 'time: %.1f ms' % (dt*1000))

				imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
				jpg = Image.fromarray(imgRGB)
				tmpFile = StringIO.StringIO()
				jpg.save(tmpFile,'JPEG')
				self.wfile.write("--jpgboundary")
				self.send_header('Content-type','image/jpeg')
				self.send_header('Content-length',str(tmpFile.len))
				self.end_headers()
				jpg.save(self.wfile,'JPEG')
				

			return


		elif (self.path == "/face_detect/cam.mjpg"):
			self.send_response(200)
			faceCascade = cv2.CascadeClassifier("/home/abhinav/opencv-2.4.9/data/lbpcascades/lbpcascade_frontalface.xml")
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			t = clock()
			while fps:
			
				img = vs.read()
			
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
				gray = cv2.equalizeHist(gray)  
				
				faces = faceCascade.detectMultiScale(
			        gray,
			        scaleFactor=1.3,
			        minNeighbors=4,
			        minSize=(30, 30),
			        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
			    	)

				#print "Found {0} faces!".format(len(faces))
				for(x, y, w, h) in faces:
					cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

				dt = clock() - t
				t = clock()
				draw_str(img, (20, 20), 'time: %.1f ms' % (dt*1000))
			    

			    	#draw_str(img, (20, 20), time: %.1f ms' % (dt*1000))
			    
				imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
				jpg = Image.fromarray(imgRGB)
				tmpFile = StringIO.StringIO()
				jpg.save(tmpFile,'JPEG')
				self.wfile.write("--jpgboundary")
				self.send_header('Content-type','image/jpeg')
				self.send_header('Content-length',str(tmpFile.len))
				self.end_headers()
				jpg.save(self.wfile,'JPEG')
			
			return

		if (self.path == "/record/cam.mjpg"):
			self.send_response(200)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			fourcc = cv2.cv.CV_FOURCC(*'XVID')
			out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
			t = clock()
			while fps:
			
				frame = vs.read()

				dt = clock() - t
				t = clock()
				draw_str(frame, (20, 20), 'time: %.1f ms' % (dt*1000))

				out.write(frame)
				    
				img = cv2.imread('image2.jpg')
				imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
				jpg = Image.fromarray(imgRGB)
				tmpFile = StringIO.StringIO()
				jpg.save(tmpFile,'JPEG')
				self.wfile.write("--jpgboundary")
				self.send_header('Content-type','image/jpeg')
				self.send_header('Content-length',str(tmpFile.len))
				self.end_headers()
				jpg.save(self.wfile,'JPEG')
				
			return

		elif self.path.endswith('.html'):
			self.send_response(200)
			self.send_header('Content-type','text/html')
			self.end_headers()
			self.wfile.write('<html><head></head><body>')
			self.wfile.write('<img src="http://127.0.0.1:8080/cam.mjpg"/>')
			self.wfile.write('</body></html>')
			return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
#class ThreadedHTTPServer(HTTPServer):
    """Handle requests in a separate thread."""


def main():
	
	try:
		server = ThreadedHTTPServer(('',8080),CamHandler)
		print "server started"
		server.serve_forever()
	except KeyboardInterrupt:
		server.socket.close()

if __name__ == '__main__':
	main()

