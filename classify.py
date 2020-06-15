# USO
# python classify.py --model kitchen.model --labelbin lb.pickle --image test/.png
# --dataset: contem o nosso modelo serializado do tipo KERAS Convolutional Neural Network (Keras CNN)
# --labelbin: ficheiro que contem o LabelBinarizer para o modelo, contem a conversão das nossas classes de Strings para um vetor de int's
# --image: imagem a analisar


from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construtor de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# carrega a imagem e faz uma cópia chamada output para fazer display
image = cv2.imread(args["image"])
output = image.copy()
 
# faz o pre-processamento da imagem para esta ficar com as mesmas carateristicas das do dataset de treino
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# carrega o modelo e o label binarizer para memoria
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# classificação da imagem e criação do label a partir do lb
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]

# compara o objeto do filename com o resultado. Se for o mesmo correct, caso contrario incorrect
#filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
#correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# constroi a label e coloca-a na imagem
#label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
label = "{}: {:.2f}%".format(label, proba[idx] * 100,)
output = imutils.resize(output, width=400) #resize para caber no ecrâ
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# mostrar a imagem com a label
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)