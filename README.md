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
├── dados/
│   ├── natural/classe_natural.txt
│   └── artificial/classe_artificial.txt
│
├── preprocessamento.py    # Passo 1: Limpeza, tokenização e estratificação
├── modelos.py             # Passo 2: Treinamento de todos os classificadores
├── main.py                # Passo 3: O arquivo mestre que chama tudo
├── requirements.txt       # Dependências (pandas, scikit-learn, spacy, nltk)
└── README.md              # Documentação
