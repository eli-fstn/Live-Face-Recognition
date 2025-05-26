import pathlib
import cv2 as cv

cascade_path = pathlib.Path(cv.__file__).parent / "data/haarcascade_frontalface_default.xml"
clf = cv.CascadeClassifier(str(cascade_path))

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces_react = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces_react:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        label = "Face"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = .5
        thickness = 1
        label_size, _ = cv.getTextSize(label, font, font_scale, thickness)
        label_width, label_height = label_size

        cv.rectangle(frame, (x, y - label_height - 10), (x + label_width + 10, y), (0, 255, 0), cv.FILLED)
        cv.putText(frame, label, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness)

    cv.imshow('Live Face Recognition', frame)

    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()