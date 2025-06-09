from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import os
from PIL import Image
import base64
import io

app = Flask(__name__)
app.secret_key = 'face_secret_key'

UPLOAD_FOLDER = 'fotos_treino'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}
label_counter = 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_base64_image(data_url, save_path):
    header, encoded = data_url.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img.save(save_path)

def train_model():
    global faces, labels, label_map, label_counter
    faces = []
    labels = []
    label_map = {}
    label_counter = 0
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            nome = fname.split('_')[0].lower()
            if nome not in label_map:
                label_map[nome] = label_counter
                label_counter += 1
            path = os.path.join(UPLOAD_FOLDER, fname)
            img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            faces_detected = face_cascade.detectMultiScale(img_gray, 1.3, 5)
            for (x, y, w, h) in faces_detected:
                face = img_gray[y:y+h, x:x+w]
                faces.append(face)
                labels.append(label_map[nome])
                break
    if faces:
        recognizer.train(faces, np.array(labels))
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    if request.method == 'POST':
        nome = request.form['nome'].strip().lower()
        # Não aceita upload, só fotos da webcam
        webcam_photos = request.form.getlist('webcam_photo')
        webcam_photos = [w for w in webcam_photos if w and w.startswith('data:image')]
        num_fotos_existentes = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(nome + '_')])
        total_fotos = len(webcam_photos) + num_fotos_existentes
        if not nome or total_fotos < 4:
            flash('Tire pelo menos 4 fotos pela câmera para cadastro.')
            return redirect(request.url)
        # Salva todas as fotos da webcam com o nome do usuário
        for j, webcam_photo in enumerate(webcam_photos):
            filename = f"{nome}_{num_fotos_existentes + j}.jpg"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            save_base64_image(webcam_photo, path)
        treinado = train_model()
        if treinado:
            flash('Usuário cadastrado e modelo treinado com sucesso!')
        else:
            flash('Nenhum rosto detectado para cadastramento.')
        return redirect(url_for('index'))
    return render_template('cadastro.html')

@app.route('/verificar', methods=['GET', 'POST'])
def verificar():
    resultado = None
    if request.method == 'POST':
        webcam_photo = request.form.get('webcam_photo')
        foto = request.files.get('foto')
        img = None
        if webcam_photo and webcam_photo.startswith('data:image'):
            header, encoded = webcam_photo.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        elif foto and allowed_file(foto.filename):
            img = Image.open(foto.stream).convert('RGB')
        if img is not None:
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            rostos = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(rostos) > 0 and faces:
                inv_label_map = {v: k for k, v in label_map.items()}
                limiar_confianca = 60
                for (x, y, w, h) in rostos:
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))
                    label, confidence = recognizer.predict(face)
                    nome_predito = inv_label_map.get(label, 'Desconhecido')
                    if confidence < limiar_confianca:
                        resultado = f"Acesso Liberado para {nome_predito} (confiança: {confidence:.1f})"
                    else:
                        resultado = f"Acesso Negado para {nome_predito} (confiança: {confidence:.1f})"
                    break
            else:
                resultado = 'Nenhum rosto detectado.'
        else:
            resultado = 'Arquivo inválido.'
    return render_template('verificar.html', resultado=resultado)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    train_model()
    app.run(debug=True)
