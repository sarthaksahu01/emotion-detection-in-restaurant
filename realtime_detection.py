# import cv2
# from keras.models import model_from_json
# import numpy as np

# json_file = open("emotionDetector.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)

# model.load_weights("emotionDetector.h5")
# haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(haar_file)


# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1, 48, 48, 1)
#     return feature / 255.0


# webcam = cv2.VideoCapture(0)
# labels = {0: "disgust", 1: "happy", 2: "sad", 3: "surprise"}
# while True:
#     i, im = webcam.read()
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(im, 1.3, 5)
#     try:
#         for (p, q, r, s) in faces:
#             image = gray[q : q + s, p : p + r]
#             cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
#             image = cv2.resize(image, (48, 48))
#             img = extract_features(image)
#             pred = model.predict(img)
#             prediction_label = labels[pred.argmax()]
#             cv2.putText(
#                 im,
#                 "% s" % (prediction_label),
#                 (p - 10, q - 10),
#                 cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                 2,
#                 (0, 0, 255),
#             )
#         cv2.imshow("Output", im)
#         cv2.waitKey(27)
#     except cv2.error:
#         pass

# import cv2
# print(cv2.__file__)


# Above is for webcam

# import cv2
# from keras.models import model_from_json
# import numpy as np

# json_file = open("emotionDetector.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)

# model.load_weights("emotionDetector.h5")
# haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(haar_file)


# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1, 48, 48, 1)
#     return feature / 255.0


# hap = 0
# sa = 0
# dis = 0
# sur = 0
# rating = 0

# webcam = cv2.VideoCapture(0)
# labels = {0: "disgust", 1: "happy", 2: "sad", 3: "surprise"}
# while True:
#     i, im = webcam.read()
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(im, 1.3, 5)
#     try:
#         for (p, q, r, s) in faces:
#             image = gray[q : q + s, p : p + r]
#             cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
#             image = cv2.resize(image, (48, 48))
#             img = extract_features(image)
#             pred = model.predict(img)
#             prediction_label = labels[pred.argmax()]
#             if prediction_label == "happy":
#                 hap += 1
#             if prediction_label == "sad":
#                 sa += 1
#             if prediction_label == "disgust":
#                 dis += 0.5
#             if prediction_label == "surprise":
#                 sur += 0.5
#             print(hap, sa, dis, sur)
#             rating = ((hap + sur) / (hap + sa + sur + dis)) * 10
#             print("Rating is", rating)
#             cv2.putText(
#                 im,
#                 "% s" % (prediction_label),
#                 (p - 10, q - 10),
#                 cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                 2,
#                 (0, 0, 255),
#             )

#         cv2.imshow("Output", im)
#         cv2.waitKey(27)
#     except cv2.error:
#         pass


# import cv2

# print(cv2.__file__)


# import cv2
# from keras.models import model_from_json
# import numpy as np

# json_file = open("emotionDetector.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)

# model.load_weights("emotionDetector.h5")
# haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(haar_file)


# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1, 48, 48, 1)
#     return feature / 255.0


# hap = 0
# sa = 0
# dis = 0
# sur = 0
# rating = 0

# video_path = "Example1.mp4"  # Update with the path to your video file
# video_capture = cv2.VideoCapture(video_path)
# labels = {0: "disgust", 1: "happy", 2: "sad", 3: "surprise"}

# while True:
#     ret, im = video_capture.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(im, 1.3, 5)

#     try:
#         for (p, q, r, s) in faces:
#             image = gray[q : q + s, p : p + r]
#             cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
#             image = cv2.resize(image, (48, 48))
#             img = extract_features(image)
#             pred = model.predict(img)
#             prediction_label = labels[pred.argmax()]

#             if prediction_label == "happy":
#                 hap += 1
#             elif prediction_label == "sad":
#                 sa += 1
#             elif prediction_label == "disgust":
#                 dis += 0.5
#             elif prediction_label == "surprise":
#                 sur += 0.5

#             rating = ((hap + sur) / (hap + sa + sur + dis)) * 10
#             print("Rating is", rating)

#             cv2.putText(
#                 im,
#                 "% s" % (prediction_label),
#                 (p - 10, q - 10),
#                 cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                 2,
#                 (0, 0, 255),
#             )

#         cv2.imshow("Output", im)
#         if cv2.waitKey(27) & 0xFF == ord("q"):
#             break
#     except cv2.error:
#         pass

# video_capture.release()
# cv2.destroyAllWindows()


import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("emotionDetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotionDetector.h5")
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


hap = 0
sa = 0
dis = 0
sur = 0
rating = 0

video_path = "Example2.mp4"
video_capture = cv2.VideoCapture(video_path)


if not video_capture.isOpened():
    print("Error: Unable to open video file.")
    exit()


initial_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
initial_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

desired_display_width = 800


aspect_ratio = initial_width / initial_height if initial_height != 0 else 1

desired_display_height = int(desired_display_width / aspect_ratio)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, initial_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, initial_height)

labels = {0: "disgust", 1: "happy", 2: "sad", 3: "surprise"}

while True:
    ret, im = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        for (p, q, r, s) in faces:
            image = gray[q : q + s, p : p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            if prediction_label == "happy":
                hap += 1
            elif prediction_label == "sad":
                sa += 1
            elif prediction_label == "disgust":
                dis += 0.5
            elif prediction_label == "surprise":
                sur += 0.5

            rating = ((hap + sur) / (hap + sa + sur + dis)) * 10
            print("Rating is", rating)

            cv2.putText(
                im,
                "% s" % (prediction_label),
                (p - 10, q - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 0, 255),
            )

        display_frame = cv2.resize(im, (desired_display_width, desired_display_height))
        cv2.imshow("Output", display_frame)

        if cv2.waitKey(27) & 0xFF == ord("q"):
            break
    except cv2.error:
        pass

video_capture.release()
cv2.destroyAllWindows()
