# SUPER-RESOLUÇÃO DE IMAGENS EM TOMOGRAFIA COMPUTADORIZADA DE BAIXA DOSAGEM: COMPARAÇÃO DE MÉTODOS DE APRENDIZADO PROFUNDO (For the english version, please refer to [this link](#super-resolution-of-low-dose-ct-images-comparison-of-deep-learning-methods))

## Índice
1. [Introdução](#introdução)
2. [Descrição do conjunto de dados](#descrição-do-conjunto-de-dados)
3. [Modelos de super-resolução utilizados](#modelos-utilizados-para-a-super-resolução)
4. [Estrutura do projeto](#estrutura-do-projeto)
5. [Como executar](#como-executar)
6. [Métricas de avaliação](#métricas-de-avaliação)
7. [Contribuições](#contribuições)

## Introdução

Este projeto visa melhorar imagens de tomografia computorizada (CT) de baixa dose utilizando métodos de **super-resolução** baseados em aprendizado profundo. As técnicas de super-resolução permitem a reconstrução de imagens de alta qualidade a partir de tomografias de baixa dose, que são tipicamente ruidosas e menos detalhadas. O projeto compara vários modelos de ponta para a super-resolução de imagens de propósito geral, utilizando uma combinação de métricas perceptivas e quantitativas para determinar o melhor desempenho.

## Descrição do conjunto de dados

O conjunto de dados está estruturado da seguinte forma:
- **Observação de TC de baixa dose**: Imagens de baixa dose e com ruído que servem de entrada.
- **Imagens de TC de alta dose**: Imagens de CT de alta qualidade utilizadas como padrão de referência.
- **Imagens super-resolvidas**: Imagens geradas por vários modelos de super-resolução.

As imagens são organizadas (após o processamento) em diretórios:
- `Dataset/observation_test/`: Imagens de TC "brutas" de baixa dose.
- `Dataset/ground_truth_test/`: Imagens de referência de TC "brutas" de dose completa.
- `Dataset/observation_test_images/`: Imagens de TC de baixa dose após conversão pelo nosso script.
- `Dataset/ground_truth_test_images/`: Imagens de referência de CT “em bruto” de dose completa após conversão pelo nosso script.
- `Imagens/MODELO/`: Contém imagens super-resolvidas geradas por cada modelo (por exemplo, SRCNN, ESRGAN, etc.).

## Modelos utilizados para a super-resolução

Os modelos de aprendizado profundo utilizados neste estudo incluem:

1. **SRCNN (Super-Resolution Convolutional Neural Network))**: Um modelo inicial de super-resolução baseado em CNN.
2. **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)**: Conhecida por gerar imagens nítidas e realistas.
3. **SwinIR (Swin Transformer for Image Restoration)**: Uma abordagem baseada em *transformers* que se centra em caraterísticas locais e globais.
4. **HAT (Hybrid Attention Transformer)**: Combina esquemas de atenção ao canal e de auto-atenção baseados em janelas.
5. **DAT (Dual Aggregation Transformer)**: O DAT agrega caraterísticas nas dimensões espacial e de canal, de forma dupla inter-bloco e intra-bloco.

## Estrutura do projeto

- `Dataset/`: Contém imagens de observação e de *ground truth*.
- `Models/`: Contém os arquivos de peso dos modelos de aprendizado profundo.
- `Images/`: Contém as imagens após serem processadas pelo chaiNNer.

## Como executar

#### **Passo 1: Requisitos de instalação**: 

Para executar o código deste repositório, é necessário instalar os seguintes pacotes:

- h5py 
- numpy 
- matplotlib 
- scikit-learn
- scikit-image
- onnx
- torch
- lpips
- pyiqa
- seaborn

Instale-os (e seus requisitos) usando pip, ou: 

```bash
pip install -r requirements.txt
```

#### **Passo 2: Preparação dos dados**

Os dados utilizados neste trabalho foram baixados do site [Zenodo](https://zenodo.org/records/3384092).

Foram utilizados dois arquivos principais deste site:

- observation_test.zip ([Link](https://zenodo.org/records/3384092/files/observation_validation.zip?download=1))
- ground_truth_test.zip ([Link](https://zenodo.org/records/3384092/files/ground_truth_test.zip?download=1))

1. Baixe os dois ficheiros mencionados
2. Descompacte-os na pasta Dataset/. Os arquivos devem estar no seguinte caminho: ./Dataset/observation_test/xxxxx.hdf5 e Dataset/ground_truth_test/xxxxx.hdf5


#### **Etapa 3: Preparação dos modelos de aprendizado profundo**

Neste trabalho, foram utilizados cinco modelos pré-treinados de aprendizado profundo, juntamente com [chaiNNer](https://chainner.app/download).

1. O primeiro modelo é o SRCNN, que pode ser baixado diretamente [deste link](https://www.dropbox.com/s/rxluu1y8ptjm4rn/srcnn_x2.pth?dl=0). Se o download falhar, vá [neste repositório](https://github.com/yjn870/SRCNN-pytorch) e escolha o modelo denominado “9-5-5”, para o fator de escala 2, que está listado na seção “Teste”.
2. O segundo modelo é o HAT, que pode ser baixado diretamente  [deste link](https://drive.google.com/file/d/16xtMezHvckdWEuSiOxcO-dgOlsI0rEUg/view?usp=drive_link). Se o download falhar, vá [neste repositório](https://github.com/chaiNNer-org/spandrel#model-architecture-support), escolha o *link* “HAT | Models” e baixe o arquivo “HAT-L_SRx2_ImageNet-pretrain.pth”. 
3. O terceiro modelo é o Real ESRGAN, que pode ser baixado diretamente [deste link](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth). Se o download falhar, vá [neste repositório](https://github.com/chaiNNer-org/spandrel#model-architecture-support), escolha o link “Real-ESRGAN Compact (SRVGGNet) | Models” e baixe o arquivo “RealESRGAN_x2plus.pth”. 
4. O quarto modelo é o SwinIR, que pode ser baixado diretamente [deste link](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth). Se o download falhar, vá [neste repositório](https://github.com/chaiNNer-org/spandrel#model-architecture-support), escolha o link “SwinIR | Models” e baixe o arquivo “001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth”. 
5. O quinto modelo é o DAT, que pode ser baixado diretamente [deste link](https://drive.google.com/file/d/1AYfLMnIqSlOJyOGabaRI48TEJh440fsN/view?usp=drive_link). Se o download falhar, vá [neste repositório](https://github.com/chaiNNer-org/spandrel#model-architecture-support), escolha o link “DAT | Models” e baixe o arquivo “DAT_x2.pth”. 

Faça o download dos arquivos com os pesos dos modelos de aprendizado profundo e coloque-os na pasta Models/.

#### **Etapa 4: Preparação dos dados e do chaiNNer**

Execute o notebook [Part 1 - Data and chaiNNer preparation](./Part%201%20-%20Data%20and%20chaiNNer%20preparation%20.ipynb) para preparar os conjuntos de dados e os modelos para executar o chaiNNer.

#### **Etapa 5: processar as imagens usando o chaiNNer**

Primeiro, devemos editar o arquivo de modelo do chaiNNer para executá-lo corretamente. Edite o arquivo [edit_chaiNNer_file.py](./edit_chaiNNer_file.py), definindo corretamente as três variáveis a seguir:

- dataset_folder
- images_folder
- models_folder

Defina-as usando o **caminho completo** para a pasta Dataset/, Models/ e Images/. Foram fornecidos exemplos considerando os sistemas Linux e Windows.

Execute o arquivo “edit_chaiNNer_file.py” para gerar o arquivo chaiNNer (“CT Super Resolution_edited.chn”) com os caminhos corretos de acordo com seu ambiente.

Abra o [chaiNNer](https://chainner.app/download) (faça o download e extraia/instale-o, caso ainda não o tenha feito). 

Carregue o arquivo gerado “CT Super Resolution_edited.chn” no chaiNNer. Instale as dependências se o chaiNNer pedir para fazer isso. 

Como editamos o arquivo fora do chaiNNer (usando o script “edit_chaiNNer_file.py”), ele emitirá um aviso. Você pode ignorá-lo.

Se tudo estiver pronto, você pode clicar no botão verde “play” na parte superior da janela do chaiNNer. Ele processará cerca de 3.500 imagens usando 5 métodos de aprendizagem profunda, e portanto, levará algum tempo. 

#### **Etapa 6: Avaliação das imagens**

Execute o notebook [Part 2 - Image Evaluation](./Part%202%20-%20Image%20Evaluation.ipynb).


## Métricas de avaliação

As métricas a seguir são usadas para avaliar o desempenho de cada modelo de super-resolução:

1. **PSNR (Peak Signal-to-Noise Ratio, relação sinal-ruído de pico)**: Mede a qualidade de reconstrução das imagens de super-resolução em comparação com o *ground truth*. É medido em dB. Valores mais altos indicam melhor qualidade.
   
2. **SSIM (Índice de Similaridade Estrutural)**: Avalia a similaridade perceptual entre as imagens super-resolvidas e as imagens "verdadeiras", com foco em luminância, contraste e estrutura. Valores mais altos indicam maior similaridade.
   
3. **LPIPS (Learned Perceptual Image Patch Similarity)**: Uma métrica perceptual que compara características profundas (geralmente extraídas em camadas profundas de redes neurais) entre duas imagens. Valores mais baixos indicam maior similaridade perceptual.

4. **RNQE (Natural Image Quality Evaluator)**: Uma métrica de qualidade de imagem sem referência. Valores mais baixos indicam melhor naturalidade e menos distorção perceptual.

5. **NRQM (Non-Reference Quality Metric, métrica de qualidade sem referência)**: Métrica sem referência usada para avaliar a qualidade das imagens sem a necessidade de uma imagem de referência. Valores mais altos indicam melhor qualidade de imagem.

6. **IPI (Índice Perceptual)**: Combina NIQE e NRQM em um único índice de qualidade perceptual. Valores mais baixos representam melhor qualidade de imagem percebida.


## Visualização e comparação

Um dos principais recursos do projeto é a comparação visual dos resultados do modelo com as imagens de observação e *ground truth*. A comparação é exibida em um *grid* que inclui:

- **Observação de TC de baixa dose**: A imagem de TC de baixa dose, usada como entrada entrada dos modelos de aprendizado profundo.
- **Imagens super-resolvidas**: Saídas dos modelos (SRCNN, ESRGAN, SwinIR, HAT, DAT).
- **Imagens de TC de alta dose**: A imagem de TC de alta qualidade e dose completa.

Todos os gráficos de comparação são salvos como arquivos PNG de alta resolução e um é exibido interativamente para cada caso de teste.

## Contribuições

Sinta-se à vontade para fazer um *fork* deste projeto e enviar solicitações para quaisquer melhorias, modelos adicionais ou correções de bugs. Contribuições, problemas e solicitações de recursos são bem-vindos.


# SUPER-RESOLUTION OF LOW DOSE CT IMAGES: COMPARISON OF DEEP LEARNING METHODS

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Models Used for Super Resolution](#models-used-for-super-resolution)
4. [Project Structure](#project-structure)
5. [How to Run](#how-to-run)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Contributions](#contributions)

## Introduction

This project aims to enhance low-dose computed tomography (CT) images using deep learning-based **super-resolution** methods. Super-resolution techniques allow for the reconstruction of high-quality images from low-dose CT scans, which are typically noisy and less detailed. The project compares several state-of-the-art models for general-purpose image super-resolution, using a combination of both perceptual and quantitative metrics to determine the best performance.

## Dataset Description

The dataset is structured as follows:
- **Observation CT Images**: Low-dose, noisy images serving as input.
- **Ground Truth CT Images**: High-quality CT images used as the reference standard.
- **Super-resolved Images**: Images generated by various super-resolution models.

The images are organized (after processing) in directories:
- `Dataset/observation_test/`: Low-dose "raw" CT images.
- `Dataset/ground_truth_test/`: Full-dose "raw" CT reference images.
- `Dataset/observation_test_images/`: Low-dose CT images after conversion by our script.
- `Dataset/ground_truth_test_images/`: Full-dose "raw" CT reference images after conversion by our script.
- `Images/MODEL/`: Contains super-resolved images generated by each model (e.g., SRCNN, ESRGAN, etc.).

## Models Used for Super Resolution

The deep learning models used in this study include:

1. **SRCNN (Super-Resolution Convolutional Neural Network)**: An early CNN-based super-resolution model.
2. **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)**: Known for generating sharp, realistic images.
3. **SwinIR (Swin Transformer for Image Restoration)**: A transformer-based approach focusing on both local and global features.
4. **HAT (Hybrid Attention Transformer)**: It combines both channel attention and window-based self-attention schemes.
5. **DAT (Dual Aggregation Transformer)**: DAT aggregates features across spatial and channel dimensions, in the inter-block and intra-block dual manner.

## Project Structure

- `Dataset/`: Contains observation and ground-truth images. 
- `Models/`: Contains the Deep Learning model weight files.
- `Images/`: Contains the images after they were processed by chaiNNer.

## How to Run

#### **Step 1: Install Requirements**: 

To run the code from this repository, you must install the following packages:

- h5py 
- numpy 
- matplotlib 
- scikit-learn
- scikit-image
- onnx
- torch
- lpips
- pyiqa
- seaborn

Install them (and their requirements) using pip, or: 

```bash
pip install -r requirements.txt
```

#### **Step 2: Data Preparation**

The data used in this work was downloaded from [Zenodo](https://zenodo.org/records/3384092).

Two files were used:

- observation_test.zip ([Link](https://zenodo.org/records/3384092/files/observation_validation.zip?download=1))
- ground_truth_test.zip ([Link](https://zenodo.org/records/3384092/files/ground_truth_test.zip?download=1))

1. Download the two mentioned files
2. Unzip them in the Dataset/ folder. The files should be in the following path: ./Dataset/observation_test/xxxxx.hdf5 and Dataset/ground_truth_test/xxxxx.hdf5


#### **Step 3: Model Preparation**

Five pre-trained deep learning models were used in this work, along with [chaiNNer](https://chainner.app/download).

1. The first model is the SRCNN, which can be downloaded directly from [this link](https://www.dropbox.com/s/rxluu1y8ptjm4rn/srcnn_x2.pth?dl=0). If the link fails, you can go to this [this repository](https://github.com/yjn870/SRCNN-pytorch) to download it. You have to choose the model named "9-5-5", for scale factor 2, which is linked in the "Test" section.
2. The second model is the HAT, which can be downloaded directly from [this link](https://drive.google.com/file/d/16xtMezHvckdWEuSiOxcO-dgOlsI0rEUg/view?usp=drive_link). If the link fails, you can go to this [this repository](https://github.com/chaiNNer-org/spandrel#model-architecture-support) to download it. You have to choose the "HAT | Models" link and download the "HAT-L_SRx2_ImageNet-pretrain.pth" file. 
3. The third model is the Real ESRGAN, which can be downloaded directly from [this link](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth). If the link fails, you can go to this [this repository](https://github.com/chaiNNer-org/spandrel#model-architecture-support) to download it. You have to choose the "Real-ESRGAN Compact (SRVGGNet) | Models" link and download the "RealESRGAN_x2plus.pth" file. 
4. The fourth model is the SwinIR, which can be downloaded directly from [this link](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth). If the link fails, you can go to this [this repository](https://github.com/chaiNNer-org/spandrel#model-architecture-support) to download it. You have to choose the "SwinIR | Models" link and download the "001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth" file. 
5. The fifth model is the DAT, which can be downloaded directly from [this link](https://drive.google.com/file/d/1AYfLMnIqSlOJyOGabaRI48TEJh440fsN/view?usp=drive_link). If the link fails, you can go to this [this repository](https://github.com/chaiNNer-org/spandrel#model-architecture-support) to download it. You have to choose the "DAT | Models" link and download the "DAT_x2.pth" file. 

Download the weight files and put in the Models/ folder.

#### **Step 4: Data and chaiNNer preparation**

Run the [Part 1 - Data and chaiNNer preparation](./Part%201%20-%20Data%20and%20chaiNNer%20preparation%20.ipynb) notebook to prepare the datasets and the models to run chaiNNer.

#### **Step 5: Process the images using chaiNNer**

First, we have to edit the chaiNNEr template file to run it. Edit the [edit_chaiNNer_file.py](./edit_chaiNNer_file.py) file, defining the three following variables properly:

- dataset_folder
- images_folder
- models_folder

Define them using the **full path** to the Dataset/, Models/ and Images/ folder. Examples considering Linux and Windows systems were provided.

Run the "edit_chaiNNer_file.py" file to generate the chaiNNer file ("CT Super Resolution_edited.chn") with the correct paths according to your environment.

Open [chaiNNer](ttps://chainner.app/download) (download and extract/install it if you have not already). 

Load the generated "CT Super Resolution_edited.chn" file inside chaiNNer. Install dependencies if chaiNNer asks to do it. 

Since we edited the file outside chaiNNer (using the "edit_chaiNNer_file.py" script) it will raise a warning. You can ignore it.

If everything is ready, you can click on the green "play" button on top of chaiNNer window. It will process ~3500 images using 5 deep learning methods, so it will take some time. 

#### **Step 6: Image evaluation**

Run the [Part 2 - Image Evaluation](./Part%202%20-%20Image%20Evaluation.ipynb) notebook.

<!-- 1. **Prepare the Dataset**: Make sure the `Dataset/` folder is populated with observation, ground truth images.

2. **Execute the Cells**:

   - **Image Loading**: Loads observation, ground truth, and model-generated images.
   - **Evaluation**: Metrics like PSNR, SSIM, LPIPS, NIQE, NRQM, and PI are calculated for each model.
   - **Visualization**: Displays comparison plots between the observation image, super-resolved images, and ground truth. -->

## Evaluation Metrics

The following metrics are used to evaluate the performance of each super-resolution model:

1. **PSNR (Peak Signal-to-Noise Ratio)**: Measures the reconstruction quality of the super-resolved images compared to the ground truth. Measured in dB. Higher values indicate better quality.
   
2. **SSIM (Structural Similarity Index)**: Assesses the perceptual similarity between the super-resolved and ground truth images, focusing on luminance, contrast, and structure. Higher values indicate higher similarity.
   
3. **LPIPS (Learned Perceptual Image Patch Similarity)**: A perceptual metric that compares deep features between two images. Lower values indicate higher perceptual similarity.

4. **NIQE (Natural Image Quality Evaluator)**: A no-reference image quality metric. Lower values indicate better naturalness and less perceptual distortion.

5. **NRQM (Non-Reference Quality Metric)**: No-reference metric used to evaluate the quality of images without the need of a ground truth. Higher values indicate better image quality.

6. **PI (Perceptual Index)**: Combines NIQE and NRQM into a single perceptual quality index. Lower values represent better perceived image quality.

## Visualization and Comparison

A key feature of the project is the visual comparison of model outputs with the observation and ground truth images. The comparison is displayed in a grid that includes:

- **Observation Image**: The input low-dose CT image.
- **Super-Resolved Images**: Outputs from the models (SRCNN, ESRGAN, SwinIR, HAT, DAT).
- **Ground Truth Image**: The high-quality, full-dose CT image.

All comparison plots are saved as high-resolution PNG files and one is displayed interactively for each test case.

## Contributions

Feel free to fork this project and submit pull requests for any improvements, additional models, or bug fixes. Contributions, issues, and feature requests are welcome.
