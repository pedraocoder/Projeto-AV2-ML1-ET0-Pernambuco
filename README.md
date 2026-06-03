# Projeto-AV2-ML1-ET0-Pernambuco
Interpolação espacial dos dados do ET0 das 12 estações meteorológicas no estado de Pernambuco em 2023.

## Autor
Pedro Soares Pereria - @pedraocoder

## Disciplina 
Machine Learning 1

## Instituição
CESAR School

## Descrição
Projeto de estimativa espacial da evapotranspiração de referência (ET₀)
no estado de Pernambuco, utilizando dados de 12 estações automáticas
do INMET no ano de 2023. Aplica a equação de Penman-Monteith FAO-56 para
cálculo da ET₀ e compara dois métodos utilizados na interpolação espacial: IDW
e Random Forest.

## Estrutura do repositório
Projeto-AV2-ML1-ET0-Pernambuco/
├── README.md
├── et0_pernambuco.ipynb
└── Dados_inmet_pernambuco_2023.zip

## Como Executar
1. Abra o notebook pelo link do colab abaixo
2. Faça upload do arquivo .zip com os dados do INMET
3. Execute as células em ordem sequencial

##  Notebook
[Abrir no Google Colab](https://colab.research.google.com/drive/1aKZrD9er-x7hV7FkUDVu7ZbCWF_hlILP?usp=sharing)

##  Hipóteses do Projeto
- A ET₀ apresenta gradiente espacial claro entre o litoral e o sertão de PE?
- O Random Forest supera o IDW na interpolação espacial da ET₀?
- É possível boa precisão com apenas 12 estações de referência?
- A altitude e a longitude são as variáveis espaciais mais influentes na predição da ET₀?

##  Passo a passo do MLflow
1. `pip install mlflow scikit-learn`
2. `python -m mlflow ui --port 5000`
3. Em outro terminal: `python mlflow_et0.py`
4. Acesse http://localhost:5000
