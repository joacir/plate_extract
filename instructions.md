# crie um programa em C++ que utilize par extrair somente a placa de uma foto de um veículo seguindo as instruções:

## Obtenha o arquivo da imagem como parametro da linha de comando

## O resultado final deverá ser o nome da imagem fonte + "_placa"

## utilize a lib OpenCV para processamento da imagem 

## Utilize as recomendações do Tesseract OCR encontradas no endereço:
https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html

## Utilize também a detecão de inclinacão e alinhamento conforme recomendações encontradas em:
https://felix.abecassis.me/2011/10/opencv-rotation-deskewing/

## Instruções de compilação

É necessário ter o OpenCV instalado.

```bash
mkdir build && cd build
cmake ..
make
```

## Exemplo de uso

```bash
./plate_extractor caminho/para/imagem.jpg
# Imprime o nome do arquivo que contém apenas a placa:
# caminho/para/imagem_placa.jpg
```
