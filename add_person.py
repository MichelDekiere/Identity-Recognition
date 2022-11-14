import cv2
import os

video = cv2.VideoCapture(0)

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  # getting a haarcascade xml file
face_cascade = cv2.CascadeClassifier()  # processing it for our project

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # adding a fallback event
    print("Error loading xml file")


nameID = str(input("Enter Your Name: ")).lower()

path = 'Images/'

isExist = os.path.exists(path)

if isExist:
    print("Name Already Taken")
    nameID = str(input("Enter Your Name Again: "))
else:
    os.makedirs(path)

while True:
    ret, frame = video.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    cv2.imshow("add_person", frame)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        if cv2.waitKey(1) == ord(' '):  # klik op spatie toets om foto te maken

            name = f'./{path}/{nameID}.jpg'
            isImageTaken = cv2.imwrite(name, frame[y:y + h, x:x + w])

            print("picture taken")
            break

    if cv2.waitKey(1) == ord('q'):  # klik op 'q' toets op af te sluiten
        break


video.release()
cv2.destroyAllWindows()
