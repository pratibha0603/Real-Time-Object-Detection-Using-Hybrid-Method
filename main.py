import cv2
from cvlib.object_detection import draw_bbox
import pyttsx3

engine = pyttsx3.init('sapi5')
voice = engine.getProperty('voices')
engine.setProperty('voice', voice[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

#def speech(text):
    # Use gTTS or any other text-to-speech library here
    # For example:
    # output = gTTS(text=text, lang='en', slow=False)
    # output.save("./sounds/output.mp3")
    # playsound("./sounds/output.mp3")

video = cv2.VideoCapture(0)
labels = []

while True:
    ret, frame = video.read()
    bbox, label, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(frame, bbox, label, conf)
    cv2.imshow("Object Detection", output_image)

    for item in label:
        if item in labels:
            pass
        else:
            labels.append(item)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

i = 0
new_sentence = []

for label in labels:
    if i == 0:
        new_sentence.append(f"I found a {label}, and, ")
    else:
        new_sentence.append(f"a {label}")
    i += 1

speak(" ".join(new_sentence))
