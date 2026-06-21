import pandas as pd
import os
import json
import re
import nltk
import spacy

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

class LegislacaoPreprocessor:
    def __init__(self, base_path="data"):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "raw")
        self.processed_path = os.path.join(base_path, "processed")
        self.index_file = os.path.join(self.raw_path, "index.csv")
        self.output_file = os.path.join(self.processed_path, "leis_clean.csv")
        self.nlp = spacy.load("pt_core_news_sm")
        os.makedirs(self.processed_path, exist_ok=True)

    def clean_text(self, text):
        text = re.sub(r"\[Detalhes da Norma\]", "", text)
        text = re.sub(r"Este texto não substitui o original publicado no Diário Oficial\.", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def process_document(self, text):
        text = self.clean_text(text)
        sentences = nltk.sent_tokenize(text, language="portuguese")
        
        cleaned_sentences = []
        all_tokens = []
        
        for sent in sentences:
            doc = self.nlp(sent)
            tokens = []
            for token in doc:
                if not token.is_punct and not token.is_space:
                    tokens.append(token.lemma_.lower())
            
            if tokens:
                cleaned_sentences.append(" ".join(tokens))
                all_tokens.extend(tokens)
                
        return text, " ".join(cleaned_sentences), sentences, all_tokens

    def run_pipeline(self):
        if not os.path.exists(self.index_file):
            return False
            
        df_index = pd.read_csv(self.index_file)
        processed_records = []
        
        for _, row in df_index.iterrows():
            text_filepath = os.path.join(self.raw_path, row["text_file"])
            if not os.path.exists(text_filepath):
                continue
                
            with open(text_filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
                
            orig_clean, text_limpo, sentencas, tokens = self.process_document(raw_text)
            
            processed_records.append({
                "id": row["id"],
                "tipo": row["tipo"],
                "numero": row["numero"],
                "ano": row["ano"],
                "data": row["data"],
                "ementa": row["ementa"],
                "meta_file": row["meta_file"],
                "texto_original": orig_clean,
                "texto_limpo": text_limpo,
                "sentencas": json.dumps(sentencas, ensure_ascii=False),
                "tokens": json.dumps(tokens, ensure_ascii=False)
            })
            
        if processed_records:
            df_out = pd.DataFrame(processed_records)
            df_out.to_csv(self.output_file, index=False, encoding="utf-8")
            return len(processed_records)
        return 0

if __name__ == "__main__":
    preprocessor = LegislacaoPreprocessor()
    total = preprocessor.run_pipeline()
    print(f"Pré-processamento concluído. Total de leis limpas: {total}")
