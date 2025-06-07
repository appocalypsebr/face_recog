# Imports
import cv2
import numpy as np
import os
#from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
#from google.colab.output import eval_js
from base64 import b64decode
from PIL import Image
from io import BytesIO


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Função para exibir imagem
def take_photo(filename='photo.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível acessar a webcam.")
        return None
    print("Pressione 'Espaço' para tirar a foto ou 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem.")
            break
        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1)
        if key == 32:  # Espaço
            cv2.imwrite(filename, frame)
            print(f"📸 Foto salva como {filename}")
            break
        elif key == ord('q'):
            print("Cancelado pelo usuário.")
            frame = None
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

# Treinar modelo
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}
label_counter = 0

for fname in os.listdir("fotos_treino"):
    if fname.endswith(".jpg") or fname.endswith(".png"):
        nome = fname.split("_")[0].lower()
        if nome not in label_map:
            label_map[nome] = label_counter
            label_counter += 1
        path = os.path.join("fotos_treino", fname)
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        faces_detected = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        print(f"Arquivo: {fname} | Nome: {nome} | Label: {label_map[nome]} | Faces detectadas: {len(faces_detected)}")
        for (x, y, w, h) in faces_detected:
            face = img_gray[y:y+h, x:x+w]
            faces.append(face)
            labels.append(label_map[nome])
            print(f"  -> Face adicionada para {nome} (label {label_map[nome]}) | shape: {face.shape}")
            break  # Apenas o primeiro rosto por imagem
print(f"labels finais: {labels}")
print(f"label_map final: {label_map}")

if faces:
    recognizer.train(faces, np.array(labels))
    print("✅ Usuários cadastrados com sucesso.")
else:
    print("⚠️ Nenhum rosto detectado para cadastramento.")

# Função para tirar foto com detecção de rosto
def tirar_foto_com_rosto(nome_arquivo='foto.jpg'):
    while True:
        # Captura a imagem da webcam retire o comentário da linha abaixo para usar webcam
        #img = take_photo(nome_arquivo)
        print("Aguardando captura de imagem...")
        # Utilizar imagem já capturada para teste
        # Se estiver rodando em um ambiente que não suporta webcam, use uma imagem de teste
        # retire o comentário da linha abaixo para usar arquivo de teste
        img = cv2.imread(nome_arquivo)
        if img is None:
            print("❌ Erro ao capturar imagem.")
            return None, None, []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rostos = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(rostos) > 0:
            print("✅ Rosto detectado.")
            return img, gray, rostos
        else:
            print("⚠️ Nenhum rosto detectado. Tente novamente.")

# Função para verificar acesso
def verificar_acesso():
    img_test, gray_test, faces_test = tirar_foto_com_rosto("teste.jpg")
    if img_test is None or gray_test is None or faces_test is None or len(faces_test) == 0:
        print("❌ Não foi possível capturar ou detectar rosto para verificação.")
        return
    inv_label_map = {v: k for k, v in label_map.items()}
    limiar_confianca = 40
    for (x, y, w, h) in faces_test:
        face = gray_test[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        label, confidence = recognizer.predict(face)
        nome_predito = inv_label_map.get(label, f"label_{label}")
        print(f"Rosto detectado: {label} com confiança {confidence:.2f}")
        print(f"Nome Predito: {nome_predito}")
        # Mostra todos os labels e nomes cadastrados para debug
        print(f"label_map: {label_map}")
        print(f"inv_label_map: {inv_label_map}")
        if confidence < limiar_confianca:
            texto = f"{nome_predito} ({confidence:.1f}) - Acesso Liberado"
            cor = (0, 255, 0)
            print(f"✅ Acesso Liberado: {nome_predito} ({confidence:.1f})")
        else:
            texto = f"{nome_predito} ({confidence:.1f}) - Acesso Negado"
            cor = (0, 0, 255)
            print(f"⚠️ Acesso Negado: {nome_predito} ({confidence:.1f})")
        cv2.rectangle(img_test, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(img_test, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
    cv2.imshow('Resultado', img_test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#Verificar o acesso*:
verificar_acesso()