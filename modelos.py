#importar bibliotecas
import pandas as pd
import spacy
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1 - Modelo baseado em estatísticas simples
def rodar_estatisticas_simples(train_df, pretest_df):
    def calcular_estatisticas(sentencas):
        num_chars = sum(len(s) for s in sentencas)
        num_words = sum(len(s.split()) for s in sentencas)
        num_sents = len(sentencas)
        num_vocab = len(set(w.lower() for s in sentencas for w in s.split()))
        
        m_chars = round(num_chars / num_words, 2) if num_words > 0 else 0
        m_sents = round(num_words / num_sents, 2) if num_sents > 0 else 0
        m_vocab = round(num_words / num_vocab, 2) if num_vocab > 0 else 0
        return m_chars, m_sents, m_vocab

    for df_temp in [train_df, pretest_df]:
        df_temp['media_caracteres_por_palavra'] = df_temp['Sentenca'].apply(lambda x: calcular_estatisticas([x])[0])
        df_temp['media_palavras_por_sentenca'] = df_temp['Sentenca'].apply(lambda x: calcular_estatisticas([x])[1])
        df_temp['media_palavras_por_palavra_unica'] = df_temp['Sentenca'].apply(lambda x: calcular_estatisticas([x])[2])

    X_train = train_df[['media_caracteres_por_palavra', 'media_palavras_por_sentenca', 'media_palavras_por_palavra_unica']]
    X_test = pretest_df[['media_caracteres_por_palavra', 'media_palavras_por_sentenca', 'media_palavras_por_palavra_unica']]
    
    # NB
    model_nbes = MultinomialNB()
    model_nbes.fit(X_train, train_df['Classe'])
    y_pred_nb = model_nbes.predict(X_test)
    print(f"-> Acurácia (NB + Estatísticas Simples): {accuracy_score(pretest_df['Classe'], y_pred_nb) * 100:.2f}%")

    # RL
    model_lres = LogisticRegression(max_iter=1000, random_state=42)
    model_lres.fit(X_train, train_df['Classe'])
    y_pred_lr = model_lres.predict(X_test)
    print(f"-> Acurácia (RL + Estatísticas Simples): {accuracy_score(pretest_df['Classe'], y_pred_lr) * 100:.2f}%")


# 2 - Modelo baseado em BoW (Bag of Words)
def rodar_bow(train_df, pretest_df):
    vectorizer = CountVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(train_df['Sentenca'])
    X_test = vectorizer.transform(pretest_df['Sentenca'])

    # NB
    model_nbbow = MultinomialNB()
    model_nbbow.fit(X_train, train_df['Classe'])
    y_pred = model_nbbow.predict(X_test)
    print(f"-> Acurácia (NB + BoW): {accuracy_score(pretest_df['Classe'], y_pred) * 100:.2f}%")

    # RL
    model_rlbow = LogisticRegression(max_iter=1000, random_state=42)
    model_rlbow.fit(X_train, train_df['Classe'])

    y_pred = model_rlbow.predict(X_test)
    print(f"-> Acurácia (RL + BoW): {accuracy_score(pretest_df['Classe'], y_pred) * 100:.2f}%")

# 3 - Modelo baseado em TF-IDF
def rodar_regressao_logistica_tfidf(train_df, pretest_df):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df['Sentenca'])
    X_test = vectorizer.transform(pretest_df['Sentenca'])

    #NB
    model_nbtfidf = MultinomialNB()
    model_nbtfidf.fit(X_train, train_df['Classe'])
    y_pred = model_nbtfidf.predict(X_test)
    print(f"-> Acurácia (NB + TF-IDF): {accuracy_score(pretest_df['Classe'], y_pred) * 100:.2f}%")

    #RL
    model_rltfidf = LogisticRegression(max_iter=1000, random_state=42)
    model_rltfidf.fit(X_train, train_df['Classe'])

    y_pred = model_rltfidf.predict(X_test)
    print(f"-> Acurácia (RL + TF-IDF): {accuracy_score(pretest_df['Classe'], y_pred) * 100:.2f}%")