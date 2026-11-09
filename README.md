# projeto-ic

# Classificador de Sentenças: Texto Natural vs. Artificial

Este projeto tem como objetivo processar, analisar e classificar sentenças em duas classes distintas (*natural* e *artificial*), utilizando técnicas de Processamento de Linguagem Natural (PLN) e algoritmos de aprendizado de máquina (Naive Bayes e Regressão Logística).

---

# Tecnologias e Bibliotecas Utilizadas
O projeto foi desenvolvido em **Python**, utilizando as seguintes bibliotecas:
* [Pandas](https://pandas.pydata.org/) (manipulação de dados)
* [Scikit-learn](https://scikit-learn.org/) (modelos de Machine Learning, vetorização e métricas)
* [NLTK](https://www.nltk.org/) (tokenização de sentenças e manipulação de texto)
* [spaCy](https://spacy.io/) (etiquetagem gramatical - POS tagging)

---

# Estrutura do Projeto
```text
classificador-sentencas-nlp/
│
├── dados/                      # Pasta com os arquivos de texto originais
│   ├── classe_natural.txt
│   └── classe_artificial.txt
│
├── main.py                     # Script principal com todo o fluxo de processamento e modelos
├── requirements.txt            # Lista de dependências do projeto
└── README.md                   # Documentação do projeto
