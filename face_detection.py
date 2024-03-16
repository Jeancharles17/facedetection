import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbor, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbor)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text , (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.0, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords, img

def detect(img, faceCascade, eyeCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
    coords, img = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Eye")
    if len(coords) ==4:
        roi_img = img[coords[1]:coords[1]+ coords[3], coords[0]:coords[0]+coords[2]]
        coords = draw_boundary(roi_img, eyeCascade, 1.1, 14, color['green'], "Face")

    return img

camera_index = 0  # Start with index 0

video_capture = cv2.VideoCapture(camera_index)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_Cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

if not video_capture.isOpened():
    print("Error: Unable to open camera.")
    exit()

while True:
    _, img = video_capture.read()
    img = detect(img, faceCascade,eye_Cascade)


    if img is None:
        print("Error: Unable to capture frame.")
        break

    cv2.imshow("face detection", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
