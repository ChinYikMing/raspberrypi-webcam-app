from flask import Flask, render_template, Response, redirect, request
from camera import USBCamera
import cv2 as cv

people_detector = cv.CascadeClassifier("fullbody_detector.xml")
#people_detector = cv.CascadeClassifier("face.xml")
camera = USBCamera()

app = Flask(__name__, template_folder='template')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/people")
def people():
    return render_template('people.html')

def responseFrame(camera):
    while True:
        ret, frame = camera.getFrame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/normalMode") # normal detection
def normalMode():
    return Response(responseFrame(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def responsePeopleFrame(camera):
    while True:
        ret, people_found, frame = camera.getPeopleFrame(people_detector)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/peopleMode") # people detection
def peopleMode():
    return Response(responsePeopleFrame(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/saveFrame")
def saveFrame():
    ret, filename = camera.saveFrame()
    if ret == True:
        return render_template('success_save.html', filename=filename)
    else:
        return render_template('fail_save.html')

@app.route("/changeNormalMode")
def changeNormalMode():
    #return redirect("http://192.168.137.27", code=302)
    return redirect("http://192.168.137.27:5000", code=302)

@app.route("/changePeopleMode")
def changePeopleMode():
    #return redirect("http://192.168.137.27/people", code=302)
    return redirect("http://192.168.137.27:5000/people", code=302)
