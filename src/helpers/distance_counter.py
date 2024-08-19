import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(1)
detector = FaceMeshDetector(maxFaces=1)
distance = None

while True:
    success, img = cap.read()
    if not success:
        break  # Handle case where the camera fails to capture an image

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[374]
        pointRight = face[145]

        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3  # The actual width of the object (in cm)

        # Finding Distance
        f = 1500  # Pre-calculated focal length
        d = (W * f) / w
        distance = int(d)  # Convert to integer
        print(distance)

        cvzone.putTextRect(img, f'Dist: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

    cv2.imshow("Image", img)

    # Break loop on key press, e.g., ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close the window
def distance_counter():
    cap = cv2.VideoCapture(1)
    detector = FaceMeshDetector(maxFaces=1)
    distance = None

    while True:
        success, img = cap.read()
        if not success:
            break  # Handle case where the camera fails to capture an image

        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            pointLeft = face[374]
            pointRight = face[145]

            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3  # The actual width of the object (in cm)

            # Finding Distance
            f = 1500  # Pre-calculated focal length
            d = (W * f) / w
            distance = int(d)  # Convert to integer
            print(distance)

        cv2.imshow("Image", img)

        # Break loop on key press, e.g., ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close the window

    return distance
