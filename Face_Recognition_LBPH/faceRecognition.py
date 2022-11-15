import numpy as np
import cv2
import os


#Face detection is done
def faceDetection(test_img):             
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    #here we give a path to the haarcascade classifier where we save in a folder.Here we detect face so we gave a frontalface_alt.xml file
    face_haar=cv2.CascadeClassifier(r'C:\Users\Acer\Documents\Explain_Project\Face_Recognition_LBPH\haarcascade_frontalface_alt.xml') 
    faces=face_haar.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=3)
    return faces,gray_img

#Labels for training data has been created

def training_data(directory):
    # initialize lists to store our extracted faces and associated
	# labels
    faces=[]
    faceID=[]
    

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system file")
                continue
            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print ("img_path",img_path)
            print("id: ",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print ("Not Loaded Properly")
                continue

            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
                continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+h,x:x+w]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID
# convert our faces and labels lists to NumPy arrays

#Here training Classifier is called
def train_classifier(faces,faceID):                              
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    # We used the cv2. face. LBPHFaceRecognizer_create to train our face recognizer on the CALTECH Faces dataset and 
    # obtained 98% accuracy
    face_recognizer.train(faces,np.array(faceID))
    # To train our LBP face recognizer, we simply call the train method, 
    # passing in our CALTECH Faces training data along with the (integer) labels for each subject.
    return face_recognizer


#Drawing a Rectangle on the Face Function
def draw_rect(test_img,face):                                      
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=3)

#Putting text on images
def put_text(test_img,text,x,y):                                    
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2,cv2.LINE_AA, False)
    

