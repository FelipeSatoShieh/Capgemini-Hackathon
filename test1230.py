import cv2
import numpy as np
import os
from PIL import Image

# # #PEGAR AS FOTOS
# def generate_dataset():
#     face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#     def face_cropped(img):
#         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, 1.3, 5)
#         # scaling factor = 1.3
#         # minimum neighbor = 5
         
#         if faces is ():
#             return None
#         for (x,y,w,h) in faces:
#             cropped_face = img[y:y+h,x:x+w]
#         return cropped_face
     
#     cap = cv2.VideoCapture(0)
#     id = 2
#     img_id = 0
     
#     while True:
#         ret, frame = cap.read()
#         if face_cropped(frame) is not None:
#             img_id+=1
#             face = cv2.resize(face_cropped(frame), (200,200))
#             face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#             file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg"
#             cv2.imwrite(file_name_path, face)
#             cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
             
#             cv2.imshow("Cropped face", face)
             
#         if cv2.waitKey(1)==13 or int(img_id)==200: #13 is the ASCII character of Enter
#             break
             
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Collecting samples is completed....")
# generate_dataset()


# #TREINAR CERTIFICADOR
# def train_classifier(data_dir):
#     path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
#     faces = []
#     ids = []

#     for image in path:
#         img = Image.open(image).convert('L')
#         imageNp = np.array(img, 'uint8')
#         id = int(os.path.split(image)[1].split(".")[1])

#         faces.append(imageNp)
#         ids.append(id)
#     ids = np.array(ids)

#     clf = cv2.face.LBPHFaceRecognizer.create()  #pip install opencv-contrib-python
#     clf.train(faces,ids)
#     clf.write("classifier.xml")
# train_classifier("data")


#DETECTAR ROSTO
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
     
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
         
        id, pred = clf.predict(gray_img[y:y+h,x:x+w])
        confidence = int(100*(1-pred/300))
         
        if confidence>82:
            if id==1:
                cv2.putText(img, "Sato", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            if id==2:
                cv2.putText(img, "Miguel", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
     
    return img
 
# loading classifier
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")
 
video_capture = cv2.VideoCapture(0)
 
while True:
    ret, img = video_capture.read()
    img = draw_boundary(img, faceCascade, 1.3, 6, (300,300,300), "Face", clf)
    cv2.imshow("face Detection", img)
     
    if cv2.waitKey(1)==13:
        break
video_capture.release()
cv2.destroyAllWindows()