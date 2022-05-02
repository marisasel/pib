<h1>Classificação de imagens de cintilografia de tireoide: BMT e doença de Graves</h1>

Implementação do trabalho de classificação de imagens de cintilografia de tireoide, proposto na disciplina de Processamento de Imagens Biomédicas. Foi fornecido um arquivo zipado com imagens no formato DICOM (Digital Imaging and Communications in Medicine) anonimizadas de 16-bits, com 128 x 128 pixels de dimensão,  de 12 pacientes - sendo seis delas de pacientes com diagnóstico de bócio multinodular tóxico (BMT) e seis de pacientes com diagnóstico de doença de Graves.

A proposta do trabalho solicitava que fosse desenvolvido um algoritmo de visão computacional para classificar a qual das duas classes a imagem do exame de cintilografia de tireoide de um determinado paciente pertence. Foi solicitado ainda o uso da estratégia "leave-one-patient-out" para fazer os testes. Tal técnica consiste em treinar o modelo do classificador com o número total de pacientes -1 e testá-lo com o paciente que ficou de fora, alternando o paciente do teste até completar a classificação de toda a amostra. Como a amostra é pequena, a especificação alertava que não seria adequado o uso de redes neurais convolucionais.


## Como rodar a versão implementada:

No terminal, primeiramente, para gerar o vetor de características, execute: 

```
$ python3 thyroid_extraction.py dataset_path/"
```

Em seguida, para testar o modelo, execute: 

```
$ python3 thyroid_model.py dataset_path/ <pre_extracted_features_file, padrão: features.txt>
```

## Entrada:
A primeira execução tem como entrada o diretório com as cintilografias, que deve conter os subdiretórios BMT e GRAVES, com subdiretórios para os pacientes e imagens ".dcm".
Na segunda execução, a entrada é o mesmo diretório da anterior, mais o arquivo de features gerado ("features.txt").

## Saída:
Ambas as versões do programa fornecem como saída impressas no terminal, acompanhando os passos de execução.
Na segunda execução, a saída são as métricas no terminal e uma janela com a relação de imagens corretas e incorretas aparece na tela.
