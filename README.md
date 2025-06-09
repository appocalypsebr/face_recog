# Reconhecimento Facial com OpenCV (LBPH)

Este projeto realiza o cadastro e reconhecimento facial de usuários utilizando Python e OpenCV, com o algoritmo LBPH (Local Binary Patterns Histograms). O sistema permite cadastrar múltiplos usuários, treinar um modelo de reconhecimento e validar o acesso por meio de uma interface web moderna, com suporte a upload de imagens e captura direta pela webcam.

## Funcionalidades
- Interface web (Flask + Bootstrap) para cadastro e verificação.
- Cadastro de usuários com múltiplas fotos (upload ou webcam).
- Treinamento automático do modelo LBPH com as imagens cadastradas.
- Reconhecimento facial em tempo real ou a partir de imagens.
- Feedback visual e textual sobre o resultado do reconhecimento.
- Suporte a múltiplos usuários.

## Estrutura do Projeto
```
app.py                # Aplicação Flask (backend e lógica)
face.py               # Script CLI de cadastro/treinamento (opcional)
verificar.py          # Script CLI de verificação (opcional)
templates/            # Templates HTML (interface web)
fotos_treino/         # Pasta com as fotos de treinamento
```

## Como Usar

### 1. Instale as dependências

```bash
pip install opencv-contrib-python numpy pillow flask
```

### 2. Execute a interface web

```bash
python app.py
```

Acesse [http://localhost:5000](http://localhost:5000) no navegador.

### 3. Cadastro de Usuários
- Clique em "Cadastrar Usuário".
- Preencha o nome e envie uma ou mais fotos (upload ou use a webcam).
- Fotos capturadas pela webcam são salvas automaticamente.
- O modelo será treinado após cada cadastro.

### 4. Verificação de Acesso
- Clique em "Verificar Acesso".
- Envie uma foto (upload) ou use a webcam para capturar uma imagem.
- O sistema exibirá o nome do usuário reconhecido e se o acesso foi liberado ou negado.

## Organização das Fotos
- As fotos de cada usuário são nomeadas no formato: `nome_0.jpg`, `nome_1.jpg`, etc.
- Exemplo:
  - `fotos_treino/diego_0.jpg`
  - `fotos_treino/elon_0.jpg`
  - `fotos_treino/steve_0.jpg`

## Observações
- O limiar de confiança (`limiar_confianca`) pode ser ajustado no código para melhorar a precisão do reconhecimento.
- Certifique-se de que as fotos estejam bem iluminadas e com o rosto centralizado.
- O modelo LBPH é robusto para pequenas variações, mas fotos de baixa qualidade podem afetar o desempenho.
- O sistema funciona em navegadores modernos que suportam acesso à webcam via JavaScript.

## Dependências
- Python 3.7+
- OpenCV (opencv-contrib-python)
- numpy
- pillow
- flask

## Licença
Este projeto é open-source e está sob a licença MIT.
