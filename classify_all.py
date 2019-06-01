# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from imutils import paths
import argparse
import imutils
import pickle
import cv2
import os
from termcolor import colored
  # load the trained convolutional neural network and the label
    # binarizer

print("[INFO] loading network...")
model = load_model("trained.model")
lb = pickle.loads(open("lb.pickle", "rb").read())



def classify_all(imagepath):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=imagepath)
    args = vars(ap.parse_args())
    
    # load the image
    image = cv2.imread(args["image"])
    output = image.copy()
     
    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # classify the input image
    print("\n[INFO] classifying image...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]

    filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
    print("[INFO] calculating for {} ".format(filename))
    #correct = "correct" if filename.rfind(label) != -1 else "incorrect"
    xx = filename.split("/")[-1].split(".")[0]
    if xx.startswith("space"):
    	xx = "space"
    if xx.startswith("nothing"):
    	xx= "nothing"
    if xx.startswith("del"):
    	xx= "del"
    else:
    	xx = xx[0:1]	
    
    correct = "correct" if xx == label else "incorrect"
    
    
    
    # build the label and draw the label on the image
    label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
    output = imutils.resize(output, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
    	0.7, (0, 255, 0), 2)
    
    # show the output image
    if correct =="correct":
        print("[INFO] {}".format(label))
    else:
        print(colored("[INFO] {}".format(label), 'red'))
imagePaths = list(paths.list_images("./dataset"))    
for  imagePath in imagePaths:
    classify_all(imagePath)
    
