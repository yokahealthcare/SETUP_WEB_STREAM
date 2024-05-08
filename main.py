import os
import threading
import time

import cv2
import imutils
import torch
from flask import Flask, request, redirect
from flask import Response
import requests


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)


@app.route("/")
def index():
    return "This is testing page. It tell you this is working :)"


@app.route("/gpu")
def gpu():
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_info += "CUDA is available. Showing GPU information:\n"
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            gpu_info += f"> GPU {i} - {gpu.name}\n"

    return f"\n{gpu_info}\n"


@app.route("/raw")
def raw():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start")
def start():
    video_input = request.args.get("video_input")

    # Start a thread that will perform motion detection
    t = threading.Thread(target=detect, args=(video_input,))
    t.daemon = True
    t.start()

    return redirect("/raw")


def detect(video_input):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock

    # Open a video source (provide the video path)
    cap = cv2.VideoCapture(video_input)


    while True:
        # Read a frame from the video source
        ret, frame = cap.read()
        # Check if the frame is successfully read
        if not ret:
            print("Error: Failed to read frame.")
            print("Restarting")
            detect(video_input)

        # Your AI
        # ....

        # Sleep
        time.sleep(0.01)
        # acquire the lock, set the output frame, and release the lock
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    while True:
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if outputFrame is None:
            continue

        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        # ensure the frame was successfully encoded
        if not flag:
            continue

        # yield the output frame in the byte format
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


# check to see if this is the main thread of execution
if __name__ == '__main__':
    """
        host : 0.0.0.0 
        - this is a must, cannot be changed to 127.0.0.1 
        - or it will cannot be accessed after been forwarded by docker to host IP
        
        port : 80 (up to you)
    """
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True, use_reloader=False)