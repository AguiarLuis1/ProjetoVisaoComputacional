# este script recolhe imagens utilizando a API Bing Image Search da Microsoft, que faz
# parte dos Serviços Cognitivos da Microsoft

# python get_images_bing.py --query "microwave" --output dataset/microwave
# --query: a categoria/nome da imagem que queremos procurar
# --output: a pasta onde vão ficar os resultados da pesquisa

from requests import exceptions
import argparse
import requests
import cv2
import os

# argumentos para a query
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
                help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory of images")
args = vars(ap.parse_args())

API_KEY = "865b447926b34db4877d725f020c7a9c"  # key da API
MAX_RESULTS = 150  # Número máximo de resultados
GROUP_SIZE = 50  # Máximo de resultados por request ao Bing API

# endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# lista de todas as exceções que podem acontecer durante o processo
# de recolha de imagens, para poderem ser filtradas
EXCEPTIONS = set([IOError, FileNotFoundError,
                  exceptions.RequestException, exceptions.HTTPError,
                  exceptions.ConnectionError, exceptions.Timeout])

# guardar o termo de pesquisa numa variavél e depois definir
# os cabeçalhos e parametros de pesquisa
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key": API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# fazer a pesquisa
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# recolha dos resultados da pesquisa (em formato json), incluindo
# a previsão do nº de resultados retornados pela API
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults,
                                                term))

# total = nº de imagens descarregadas até ao momento
total = 0

# loop over no nº de resultados estimados nos grupos de tamanho GROUP_SIZE
for offset in range(0, estNumResults, GROUP_SIZE):
    # update do parametro de pesquisa utilizando o offset atual, depois
    # faz o pedido para buscar os resultados
    print("[INFO] making request for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(
        offset, offset + GROUP_SIZE, estNumResults))

    # loop over nos resultados
    for v in results["value"]:
        # tenta fazer download da imagem
        try:
            # faz um request para fazer download da imagem
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)

            # constroi o caminho até ao output da imagem
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([args["output"], "{}{}".format(
                str(total).zfill(4), ext)])
            print(p)

            # escreve a imagem em disco
            f = open(p, "wb")
            f.write(r.content)
            f.close()

        # catch de erros que não nos permitam fazer o download da imagem
        except Exception as e:
            # verifica se a exceção está na nossa lista de exceções previstas
            if type(e) in EXCEPTIONS:
                print(e)
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue

        # tenta carregar a imagem do disco
        image = cv2.imread(p)

        # se não conseguir carregar a imagem de disco, ou seja o conteúdo
        # de image for None, esta é ignorada
        if image is None:
            print("[INFO] deleting: {}".format(p))
            os.remove(p)
            continue

        # update no contador do total de imagens
        total += 1
