# USO
# python train.py --dataset dataset --model kitchen.model --labelbin lb.pickle
# --dataset: nome do destino do modelo serializado do tipo KERAS Convolutional Neural Network (Keras CNN)
# --labelbin: nome para o ficheiro que contem o LabelBinarizer para o modelo, contem a conversão das nossas classes de Strings para um vetor de int's
# --plot: local de destino para a uma grafico com os resultados do treino

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
# configurar matplotlib no backend, para desta forma as figuras poderem ser salvas no background
import matplotlib


matplotlib.use("Agg")

# construtor de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# definir numero de "epochs", taxa de aprendizagem inicial,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3  # 1e-3 é o valor default para o otimizador Adam
BS = 32
IMAGE_DIMS = (96, 96, 3)

# inicializar a data e labels que vão guardar as imagens e labels preprocessadas, respetivamente
data = []
labels = []

# "baralha" as imagens entre elas, desta forma não se treina as imagens de cada classe sequencialmente,
# evitando o modelo de ficar "viciado"
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop para todas as imagens
for imagePath in imagePaths:
    # carrega a imagem, esta é processada e guardada na data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)  # converte a imagem para um array compativel com o Keras
    data.append(image)

    # extrair a label da imagens(a partir do path ) e adicioná-la á lista de labels
    # pega no penultimo parametro do caminho (dataset/{CLASS_LABEL}/{FILENAME}.jpg)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# converte a lista data para um array do NumPy e normaliza-se os pixels (de 0->255 para 0->1)
# converte também a lista labels para um array NumPy
# print do tamanho da matriz da data
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))

# binarize das labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# divisão da data em 80% para treino e 20% para teste
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)

# uma vez que estamos a trabalhar com um dataset reduzido devido á especificidade, utilizamos esta
# ferramenta para o aumentar. Esta, a partir das nossas imagens cria mais imagens baseadas na existentes
# para treinar o modelo
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# inicializar o modelo
# inicializar o otimizador Adam
# compilar o modelo com "categorical cross-entropy" uma vez que temos > 2 classes
print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

#categorical_crossentropy -> mais de 2
#binary_crossentropy -> 2
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# treinar a rede
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# guardar o modelo em disco
print("[INFO] serializing network...")
model.save(args["model"])

# guardar o LabelBinarizer em disco
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# cria e guarda a imagem do gráfico
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
