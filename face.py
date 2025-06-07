# Instalar OpenCV com contrib (para LBPH)
#!pip install opencv-contrib-python
#pip install ipython

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

def show_image(img, window_name="Imagem"):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Função para capturar imagem da webcam
""" 
def take_photo(filename='photo.jpg'):
    js_code = '''
        async function takePhoto() {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = '📸 Tirar Foto';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getTracks().forEach(t => t.stop());
            div.remove();

            const dataUrl = canvas.toDataURL('image/jpeg');
            return dataUrl;
        }
        takePhoto();
    '''
    data = eval_js(js_code)
    binary = b64decode(data.split(',')[1])
    img = Image.open(BytesIO(binary))
    img.save(filename)
    print(f"📸 Foto salva como {filename}")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
"""
import cv2

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

# Função para tirar foto com detecção de rosto
def tirar_foto_com_rosto(nome_arquivo='foto.jpg'):
    while True:
        img = take_photo(nome_arquivo)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rostos = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(rostos) > 0:
            print("✅ Rosto detectado.")
            return img, gray, rostos
        else:
            print("⚠️ Nenhum rosto detectado. Tente novamente.")

# Baixar classificador Haar
cascade_url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml"
#!wget -nc {cascade_url} -P .
#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Criar pasta para fotos
os.makedirs("fotos_treino", exist_ok=True)

# Cadastro de usuário
nome = input("Digite o nome da pessoa para cadastrar o usuário: ")
for i in range(10):
    print(f"Tirando foto {i+1}/10 para {nome}")
    img, gray, rostos = tirar_foto_com_rosto(f"fotos_treino/{nome}_{i}.jpg")
    cv2.imwrite(f"fotos_treino/{nome}_{i}.jpg", img)

# Treinar modelo
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {nome: 0}

for fname in os.listdir("fotos_treino"):
    path = os.path.join("fotos_treino", fname)
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    faces_detected = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        face = img_gray[y:y+h, x:x+w]
        faces.append(face)
        labels.append(label_map[nome])
        break

if faces:
    recognizer.train(faces, np.array(labels))
    print("✅ Usuário cadastrado com sucesso.")
else:
    print("⚠️ Nenhum rosto detectado para cadastramento.")

# Reconhecimento
print("\nTire uma foto para testar reconhecimento")
img_test, gray_test, faces_test = tirar_foto_com_rosto("teste.jpg")

inv_label_map = {v: k for k, v in label_map.items()}
limiar_confianca = 40
for (x, y, w, h) in faces_test:
    face = gray_test[y:y+h, x:x+w]
    face = cv2.resize(face, (200, 200))

    if faces:
        label, confidence = recognizer.predict(face)
        nome_predito = inv_label_map.get(label, "Desconhecido")

        if confidence < limiar_confianca:
            texto = f"{nome_predito} ({confidence:.1f}) - Acesso Liberado"
            cor = (0, 255, 0)
        else:
            texto = f"{nome_predito} ({confidence:.1f}) - Acesso Negado"
            cor = (0, 0, 255)

        cv2.rectangle(img_test, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(img_test, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
    else:
        cv2.rectangle(img_test, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img_test, "Modelo Nao Treinado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

show_image(img_test)

def verificar_acesso():

   img_test, gray_test, faces_test = tirar_foto_com_rosto("teste.jpg")

   inv_label_map = {v: k for k, v in label_map.items()}
   limiar_confianca = 40
   for (x, y, w, h) in faces_test:
     face = gray_test[y:y+h, x:x+w]
     face = cv2.resize(face, (200, 200))

     if faces:
        label, confidence = recognizer.predict(face)
        nome_predito = inv_label_map.get(label, "Desconhecido")
        print(f"Nome Predito: {nome_predito}")

        if confidence < limiar_confianca:
            texto = f"{nome_predito} ({confidence:.1f}) - Acesso Liberado"
            cor = (0, 255, 0)
            print("✅ Acesso Liberado")
        else:
            texto = f"{nome_predito} ({confidence:.1f}) - Acesso Negado"
            cor = (0, 0, 255)
            print("⚠️ Acesso Negado")
            print(f"{nome_predito} ({confidence:.1f}) - Acesso Negado")

        cv2.rectangle(img_test, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(img_test, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
     else:
        cv2.rectangle(img_test, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img_test, "Modelo Nao Treinado", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("⚠️ Modelo não treinado")



#Verificar o acesso*:
verificar_acesso()