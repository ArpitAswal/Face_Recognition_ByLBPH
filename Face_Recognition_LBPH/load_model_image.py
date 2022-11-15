import cv2
import draw_rectangle as dr
import faceRecognition as fr

# path is the image of which we want to test
test_img=cv2.imread(r'C:\Users\Acer\Documents\Explain_Project\Face_Recognition_LBPH\Resume_pic.jfif')    

faces_detected,gray_img=fr.faceDetection(test_img)
print("face Detected: ",faces_detected)

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#Give path of where trainingData.yml is saved
face_recognizer.read(r'C:\Users\Acer\Documents\Explain_Project\Face_Recognition_LBPH\trainingData.yml') 

resized_img=dr.rectangle(faces_detected,gray_img,face_recognizer,test_img)

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
