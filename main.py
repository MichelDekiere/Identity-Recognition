import cv2
import numpy as np
import face_recognition
import os

font = cv2.FONT_HERSHEY_SIMPLEX

path = 'Images'
images = []
classNames = []
classesList = os.listdir(path)
print(classesList)

for clss in classesList:
    image = cv2.imread(f'{path}/{clss}')
    images.append(image)
    classNames.append(os.path.splitext(clss)[0])

print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
print('-------Encoding succesful-------')

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  # getting a haarcascade xml file
face_cascade = cv2.CascadeClassifier()  # processing it for our project

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # adding a fallback event
    print("Error loading xml file")


video = cv2.VideoCapture(0)  # parameter 0 omdat we maar 1 camera hebben (1 webcam), indien bv. 2 camera's -> parameter 1, enz ...

# Check if the webcam is opened correctly
if not video.isOpened():
    raise IOError("Cannot open webcam")

while video.isOpened():  # checking if are getting video feed and using it
    ret, frame = video.read()  # ret is a Boolean value returned by the read function, and it indicates whether or not the frame was captured                                successfully. If the frame is captured correctly, it's stored in the variable frame.

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #changing the video to grayscale to make the face analisis work properly
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)  # vierkant rond gezicht zetten + kleur

                # emotie op webcam beeld afdrukken
                cv2.putText(frame,
                            name,
                            (x, y),
                            font, 1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_4)

        cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == ord('q'):  # klik op 'q' toets op af te sluiten
        break

video.release()
cv2.destroyAllWindows()

# imgElon = face_recognition.load_image_file('Images/Elon Musk.jpg')
# imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file('Images/Bill Gates.jpg')
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
