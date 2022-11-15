import cv2
import faceRecognition as fr

def rectangle(faces_detected,gray_img,face_recognizer,test_img):
    name={0:"Name=Arpit",1:"Kohli"}     #Change names accordingly. If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.
    id={0:"Id_No=000",1:"Id_No=001"}
    for face in faces_detected:
     (x,y,w,h)=face
     roi_gray=gray_img[y:y+h,x:x+w]
     prediction,confidence=face_recognizer.predict(roi_gray)
     print ("Confidence :",confidence)
     print("Prediction label :",prediction)
     # We initialize two variable, prediction and confidence,to store the predicted class label and the confidence/probability of the prediction.
     fr.draw_rect(test_img,face)
     if(confidence<80):
       if(prediction==0):
         fr.put_text(test_img,name[prediction],x,y-30)
         fr.put_text(test_img,id[prediction],x,y)
       elif(prediction==1):
         fr.put_text(test_img,name[prediction],x,y-30)
         fr.put_text(test_img,id[prediction],x,y)
         
     else:
       fr.put_text(test_img,"Name=Unknown",x,y-30)
       fr.put_text(test_img,"Id_No=Notfound",x,y)
         
     
     
    '''For each of these faces, we call the predict method of the recognizer which returns a 2-tuple of-
(1) the prediction (i.e., the integer label of the subject) and 
(2) the conf (short for confidence) which is simply the xrectangle((chi^{2}) distance between the current testing vector 
and the closest data point in the training data. 
The lower the distance, more likely the two faces are of the same subject.'''     
    resized_img=cv2.resize(test_img,(800,800))
    return resized_img