#importar bibliotecas
import re
import pandas as pd
import nltk
from sklearn.model_selection import StratifiedShuffleSplit

# função para tokenizar textos e balancear as classes
def preparar_dados():
    nltk.download('punkt_tab', quiet=True)
    
    # caminhos locais
    path_natural = 'dados/natural/classe_natural.txt'
    path_artificial = 'dados/artificial/classe_artificial.txt'

    # processar classe artificial
    with open(path_artificial, 'r', encoding='latin-1') as f:
        texto_art = f.read()
    texto_limpo_art = re.sub(r'\s+', ' ', texto_art)
    sentencas_art = nltk.sent_tokenize(texto_limpo_art)
    df_artificial = pd.DataFrame({'sentenca': sentencas_art})
    df_artificial['Classe'] = 'artificial'
    
    # extrair quantidade de sentenças da classe artificial
    total_artificial = len(df_artificial)
    print(f"Total de sentenças artificiais: {total_artificial}")

    # processar classe natural
    with open(path_natural, 'r', encoding='utf-8') as f:
        texto_nat = f.read()
    texto_limpo_nat = re.sub(r'\s+', ' ', texto_nat)
    sentencas_nat = nltk.sent_tokenize(texto_limpo_nat)
    df_natural = pd.DataFrame({'sentenca': sentencas_nat})
    df_natural = df_natural.sample(frac=1, random_state=42).iloc[7894:].reset_index(drop=True)
    
    # selecionar aleatoriamente o mesmo número de sentenças da classe artificial
    df_natural = df_natural.sample(n=total_artificial, random_state=42).reset_index(drop=True)
    df_natural['Classe'] = 'natural'
    
    print(f"Total de sentenças naturais com balanceamento): {len(df_natural)}")

    # combinar datasets
    df_total = pd.concat([df_natural, df_artificial], ignore_index=True)
    print(f"-> Dataset total balanceado: {len(df_total)} sentenças")

    # estratificação (80% treino / 20% pré-teste)
    all_sents = list(zip(df_total['sentenca'], df_total['Classe']))
    labels = [classe for (_, classe) in all_sents]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, pretest_index in split.split(all_sents, labels):
        strat_train_set = [all_sents[i] for i in train_index]
        strat_pretest_set = [all_sents[i] for i in pretest_index]

    train_df = pd.DataFrame(strat_train_set, columns=['Sentenca', 'Classe'])
    pretest_df = pd.DataFrame(strat_pretest_set, columns=['Sentenca', 'Classe'])

    return train_df, pretest_df