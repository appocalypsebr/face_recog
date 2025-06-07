# Reconhecimento Facial com OpenCV (LBPH)

Este projeto realiza o cadastro e reconhecimento facial de usuários utilizando Python e OpenCV, com o algoritmo LBPH (Local Binary Patterns Histograms). O sistema permite cadastrar múltiplos usuários, treinar um modelo de reconhecimento e validar o acesso por meio de uma webcam ou imagens de teste.

## Funcionalidades
- Cadastro de usuários com múltiplas fotos.
- Treinamento automático do modelo LBPH com as imagens cadastradas.
- Reconhecimento facial em tempo real ou a partir de imagens.
- Feedback visual e textual sobre o resultado do reconhecimento.
- Suporte a múltiplos usuários.

## Estrutura do Projeto
```
face.py           # Script de cadastro de usuários e treinamento
verificar.py      # Script de verificação/reconhecimento facial
fotos_treino/     # Pasta com as fotos de treinamento (uma subpasta por usuário)
```

## Como Usar

### 1. Instale as dependências

```bash
pip install opencv-contrib-python numpy pillow
```

### 2. Cadastro de Usuários
Execute o script de cadastro para registrar novos usuários e tirar fotos:

```bash
python face.py
```

- Siga as instruções para capturar fotos de cada usuário.
- As imagens serão salvas na pasta `fotos_treino/`.

### 3. Treinamento e Verificação
Execute o script de verificação para reconhecer usuários:

```bash
python verificar.py
```

- O script irá treinar o modelo com as imagens da pasta `fotos_treino/`.
- Para testar, tire uma foto ou use uma imagem de teste.
- O sistema exibirá o nome do usuário reconhecido e se o acesso foi liberado ou negado.

## Organização das Fotos
- As fotos de cada usuário devem ser nomeadas no formato: `nome_0.jpg`, `nome_1.jpg`, etc.
- Exemplo:
  - `fotos_treino/diego_0.jpg`
  - `fotos_treino/elon_0.jpg`
  - `fotos_treino/steve_0.jpg`

## Observações
- O limiar de confiança (`limiar_confianca`) pode ser ajustado para melhorar a precisão do reconhecimento.
- Certifique-se de que as fotos estejam bem iluminadas e com o rosto centralizado.
- O modelo LBPH é robusto para pequenas variações, mas fotos de baixa qualidade podem afetar o desempenho.

## Dependências
- Python 3.7+
- OpenCV (opencv-contrib-python)
- numpy
- pillow

## Licença
Este projeto é open-source e está sob a licença MIT.
