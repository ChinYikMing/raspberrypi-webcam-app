import cv2
from datetime import datetime
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np

class USBCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture("/dev/video0")
    
    def __del__(self):
        #releasing camera
        self.video.release()

    def getFrame(self):
        #extracting frames
        ret, frame = self.video.read()
        ret2, jpeg = cv2.imencode('.jpg', frame);
        return ret2, jpeg.tobytes()

    def getPeopleFrame(self, model):
        found_people = False
        ret, frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        people = model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(people) > 0:
            found_people = True

        for (x, y, w, h) in people:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return ret, found_people, jpeg.tobytes()

    def saveFrame(self):
        ret, frame = self.video.read()
        now = datetime.now()
        frameDir = "frames/"
        filename = now.strftime("%Y%m%d %H%M%S") + ".jpg"
        ret = cv2.imwrite(frameDir + filename, frame)
        return ret, filename
