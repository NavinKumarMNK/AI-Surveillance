from datetime import datetime
from time import sleep, time
import cv2
import numpy as np
import threading
import queue

PORT = 8555
TYPE = 265  # 264

fps = 6
width = 1920
height = 1080

# Create a queue to pass frames from the main thread to the printing thread
frame_queue = queue.Queue()

# Define a function for the printing thread to print frames with timestamps
def print_frames():
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)  # Create a window for displaying frames
    while True:
        frame = frame_queue.get()
        if frame is not None:
            cv2.imshow("Frame", frame)  # Display the frame
            cv2.waitKey(1)  # Wait for a short time to allow the frame to be displayed
        frame_queue.task_done()

# Start the printing thread
print_thread = threading.Thread(target=print_frames)
print_thread.daemon = True
print_thread.start()

try:
    # Define the GStreamer pipeline for RTSP streaming
    if TYPE == 265:
        pipeline = ('appsrc ! videoconvert ! video/x-raw,format=I420 ! x265enc speed-preset=ultrafast bitrate=600 key-int-max='
                    + str(fps * 2) + ' ! video/x-h265 ! rtspclientsink location=rtsp://192.168.1.10:'+ str(PORT) + '/mystream')
    elif TYPE == 264:
        pipeline = ('appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc speed-preset=ultrafast bitrate=600 key-int-max='
                + str(fps * 2) + ' ! video/x-h264,profile=baseline ! rtspclientsink location=rtsp://192.168.1.10:'+ str(PORT) + '/mystream')

    cap = cv2.VideoCapture(0)  # params(CAMERA_INDEX)
    
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
    if not out.isOpened():
        raise Exception("Can't open video writer")

    curcolor = 0
    start = time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Push the captured frame to the queue for printing
        frame_queue.put(frame)
        print("%s Frame captured at %s" % (frame.shape, datetime.now()))
            
        out.write(frame)  # Send frame to the RTSP server
        
        
        now = time()
        diff = (1 / fps) - now - start
        if diff > 0:
            sleep(diff)
        start = now

# Release the camera and video writer when done
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    out.release()
