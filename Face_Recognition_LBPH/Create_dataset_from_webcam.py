import cv2
#import sys
count = 0

#VideoCapture open a camera (our webcam).we can pass either index or the name of a file as a arguments
vidStream = cv2.VideoCapture(0)
while True:
    # read() returns a bool(true/false) and the frame which webcam is currently reading.
    ret, frame = vidStream.read() 
    
    # show resulting frame in window(in string is frame name,resulting frame)
    cv2.imshow("test window", frame) 
    
    #path to where we want to save our dataset and keep image%04i.jpg. Here our images will be stored at train-images/0/ folder
    cv2.imwrite(r"C:\Users\Acer\Documents\Explain_Project\Face_Recognition_LBPH\train-images\0\image"+str(count)+".jpg",frame )    
    count += 1
    
    #waitKey stop the frame window so we can see our result and until we press q it extract a frame as a image from webcam.
    if cv2.waitKey(10)==ord('q'):
        break
        

