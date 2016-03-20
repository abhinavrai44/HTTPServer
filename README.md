# HTTPServer
Implement an HTTP server on a raspberry pi module. The HTTP server is threaded (i.e. multiple users can access the video stream at any given instant) and allows you to access different functionalities such as object tracking, face recognition, finger counting, etc by accessing different links.

Different links display different functionalities :- 
1. (raspberry pi ip / pc ip):8080/cam.mjpg - Simple video streaming
2. (raspberry pi ip / pc ip):8080/finger_count/cam.mjpg - Finger Counting
3. (raspberry pi ip / pc ip):8080/track_object/cam.mjpg - Object Tracking
4. (raspberry pi ip / pc ip):8080/face_detect/cam.mjpg - Face Detection
5. (raspberry pi ip / pc ip):8080/record/cam.mjpg - Start Recoding the Video
6. (raspberry pi ip / pc ip):8080/record_end/cam.mjpg - Stop the Recording
7. (raspberry pi ip / pc ip):8080/gps_data.html - Starts transmitting the GPS data over the network

Note :-
1. vid_cam.py - Works on normal pc/raspberry pi using the webcam as the source of video input.
2. vid_rasp.py - This works only on a raspberry pi and uses raspicam for video input.
3. gps.py - Just another code wherein I am sharing the co-ordinates obtained from the gps module with another android application along with the functionalities of vid_rasp.py.
