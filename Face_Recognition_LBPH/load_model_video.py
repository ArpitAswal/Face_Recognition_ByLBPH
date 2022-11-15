import cv2
import faceRecognition as fr
import draw_rectangle as dr

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\Acer\Documents\Explain_Project\Face_Recognition_LBPH\trainingData.yml')    
cap=cv2.VideoCapture(0)  

while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)
    
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

    resized_img=dr.rectangle(faces_detected,gray_img,face_recognizer,test_img)

    cv2.imshow("face detection ", resized_img)
    if cv2.waitKey(10)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()   