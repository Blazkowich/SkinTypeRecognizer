import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(1)
detector = FaceMeshDetector(maxFaces=1)
distance = None

while True:
    success, img = cap.read()
    if not success:
        break  # handle case where the camera fails to capture an image

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[374]
        pointRight = face[145]

        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3  # the actual width of the object in cm

        # finding Distance
        f = 1500  # pre-calculated focal length - you need to calculate this according to your webcam focal length
        d = (W * f) / w
        distance = int(d)  # convert to integer
        print(distance)

        cvzone.putTextRect(img, f'Dist: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
