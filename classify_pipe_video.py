# USO
# python classify_pipe_video.py --model kitchen.model --labelbin lb.pickle --input test/car.mp4 --output test/result.avi
# --dataset: contem o nosso modelo serializado do tipo KERAS Convolutional Neural Network (Keras CNN)
# --labelbin: ficheiro que contem o LabelBinarizer para o modelo, contem a conversão das nossas classes de Strings para um vetor de int's
# --input: ficheiro com o video a analisar
# --output: local onde se pretende guardar o resultado


from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import time
import os

# construtor de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to label binarizer")
ap.add_argument("-i", "--input", required=True,
                help="path to input image")
ap.add_argument("-o", "--output", required=True,
                help="path to output video")
args = vars(ap.parse_args())

# iniciar a stream de video, um apontador para o ficheiro de output e as dimensões do frame
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# tentar determinar o nº de frames no video
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
# se ocorrer um erro enquanto tenta determinar o nº de frames
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# carrega o modelo e o label binarizer para memoria
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

coffeeModel = load_model("coffee.model")
lbCoffee = pickle.loads(open("lbCoffee.pickle", "rb").read())

stoveModel = load_model("stove.model")
lbStove = pickle.loads(open("lbStove.pickle", "rb").read())

while True:

    # ler o proximo frame do video
    (grabbed, frame) = vs.read()

    # se o frame não foi capturado, então chegamos ao fim do video
    if not grabbed:
        break

    # se ainda não se tiver as dimensões do frame, então estas são atribuidas
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # copia do frame inicial, uma vez que este vai ser modificado para a analise
    output = frame.copy()

    start = time.time()
    # faz o pre-processamento da imagem para esta ficar com as mesmas carateristicas das do dataset de treino
    frame = cv2.resize(frame, (96, 96))
    frame = frame.astype("float") / 255.0
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)

    # classificação da imagem e criação do label a partir do lb
    try:
        proba = model.predict(frame)[0]
        idx = np.argmax(proba)
        label = lb.classes_[idx]
    except:
        print("An exception occurred in prediction")
        continue

    # se detetar um fogão ou uma máquina de café, faz um pipe para cada um dos submodelos
    # lê o respetivo submodelo e lb e atribui a sublabel a label
    if label == "coffee":  # coffee
        probaCoffee = coffeeModel.predict(frame)[0]
        idxCoffee = np.argmax(probaCoffee)
        label = lbCoffee.classes_[idxCoffee]
    if label == "stove":
        probaStove = stoveModel.predict(frame)[0]
        idxStove = np.argmax(probaStove)
        label = lbStove.classes_[idxStove]

    # constroi a label e coloca-a na imagem
    label = "{}: {:.2f}%".format(label, proba[idx] * 100, )

    #dá resize á imagem
    output = imutils.resize(output, width=800)

    # escrever o eletrodomestico e a probabilidade na imagem
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # ler tutorial de utilização a partir do txt, depende da label
    labelArray = x = label.split()  # dividir por espaços
    name = labelArray[0]  # nome sem probabilidade
    name = name.replace(':', '')  # tirar os ':'
    with open('tutorials/' + name + '.txt', 'r') as myfile:
        tutorial = myfile.read()

    # variaveis usadas para escrever várias linhas (y0 - y inicial, dy - salto no y)
    y0, dy = 45, 20
    # para cada \n no ficheiro de texto escreve-se uma nova linha
    for i, line in enumerate(tutorial.split('\n')):
        y = y0 + i * dy
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 2)

    end = time.time()

    # verificar se writer é None
    if writer is None:
        # inicializar o video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (output.shape[1], output.shape[0]), True)

        # tempo de processamento de um frame e tempo estimado para tdo o video
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # escrever o frame output em disco
    writer.write(output)

# libertar os pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
