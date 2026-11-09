from preprocessamento import preparar_dados
from modelos import (
    rodar_estatisticas_simples, 
    rodar_bow, 
    rodar_regressao_logistica_tfidf
)

if __name__ == "__main__":
    print("=== INICIANDO PIPELINE DE CLASSIFICAÇÃO ===")
    
    print("\n1. Carregando e preparando os dados (tokenização e estratificação)...")
    train_df, pretest_df = preparar_dados()
    print(f"-> Treinamento: {len(train_df)} sentenças")
    print(f"-> Pré-teste: {len(pretest_df)} sentenças")

    print("\n2. Executando Modelos e Avaliações...")
    
    print("\n--- [Etapa 1] Estatísticas Simples ---")
    rodar_estatisticas_simples(train_df, pretest_df)
    
    print("\n--- [Etapa 2] Bag of Words (BoW) ---")
    rodar_bow(train_df, pretest_df)
    
    print("\n--- [Etapa 3] TF-IDF ---")
    rodar_regressao_logistica_tfidf(train_df, pretest_df)
    
    print("\n=== PROCESSO CONCLUÍDO ===")