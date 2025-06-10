---
layout: default
title: Reconhecimento Facial com OpenCV (LBPH)
---

# Reconhecimento Facial com OpenCV (LBPH)

Este projeto realiza o cadastro e reconhecimento facial de usuários utilizando Python e OpenCV, com o algoritmo LBPH (Local Binary Patterns Histograms). O sistema permite cadastrar múltiplos usuários, treinar um modelo de reconhecimento e validar o acesso por meio de uma interface web moderna, com suporte a upload de imagens e captura direta pela webcam.

## Demonstração em Vídeo

[![Veja o vídeo de demonstração no YouTube](https://img.youtube.com/vi/6t5S2ZSCIq0/0.jpg)](https://www.youtube.com/watch?v=6t5S2ZSCIq0)

## Funcionalidades
- Interface web (Flask + Bootstrap) para cadastro e verificação.
- Cadastro de usuários com múltiplas fotos pela webcam.
- Treinamento automático do modelo LBPH com as imagens cadastradas.
- Reconhecimento facial em tempo real ou a partir de imagens.
- Feedback visual e textual sobre o resultado do reconhecimento.
- Suporte a múltiplos usuários.

## Como Usar

1. Instale as dependências:
   ```bash
   pip install opencv-contrib-python numpy pillow flask
   ```
2. Execute a interface web:
   ```bash
   python app.py
   ```
3. Acesse [http://localhost:5000](http://localhost:5000) no navegador.
4. Cadastre usuários tirando pelo menos 4 fotos pela câmera.
5. Verifique o acesso tirando uma foto ou enviando uma imagem.

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
