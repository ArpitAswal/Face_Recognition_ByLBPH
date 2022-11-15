import cv2
import draw_rectangle as dr
import faceRecognition as fr

#Give path to the image which you want to test
#test_img=cv2.imread(r'C:\Users\Acer\Documents\Face-Recognition-master\Resume_pic.jpg')  
#faces_detected,gray_img=fr.faceDetection(test_img)
#print("face Detected: ",faces_detected)

#Training will begin from here

faces,faceID=fr.training_data(r'C:\Users\Acer\Documents\Explain_Project\Face_Recognition_LBPH\train-images') 
#Give path to the train-images folder which has all labeled folder as 0 and 1 and so on.
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'C:\Users\Acer\Documents\Explain_Project\Face_Recognition_LBPH\trainingData.yml') 
#It will save the trained model. Just give path to where you want to save

#resized_img=dr.rectangle(faces_detected,gray_img,face_recognizer,test_img)
#cv2.imshow("face detection ", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows
