import json
import os
import re

import pandas as pd


class LegislacaoFeatureExtractor:
    def __init__(self, base_path="data"):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "raw")
        self.processed_path = os.path.join(base_path, "processed")
        self.clean_file = os.path.join(self.processed_path, "leis_clean.csv")
        self.output_file = os.path.join(self.processed_path, "leis_features.csv")
        
        self.termos_juridicos = {
            "parágrafo", "inciso", "alínea", "promulgação", "revogação",
            "competência", "jurisdição", "decreto", "sanção", "veto",
            "vigência", "eficácia", "dispositivo", "regulamento",
            "inconstitucionalidade", "legalidade", "tutela", "ônus",
            "acórdão", "trânsito em julgado", "embargo", "agravo",
            "liminar", "petição", "réu", "autor", "procurador",
            "promotor", "juiz", "tribunal", "magistrado", "súmula",
            "código", "constituição", "ementa", "preâmbulo", "epígrafe",
            "disposições", "transitórias", "subsidiário", "incumbência",
            "atribuição", "competente", "pleito", "litígio", "homologação"
        }

    def count_syllables_pt(self, word):
        word = word.lower()
        vowels = "aeiouyáéíóúâêôãõàü"
        count = 0
        prev_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel:
                if not prev_was_vowel:
                    count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        return max(1, count)

    def calculate_flesch(self, words, num_sentences):
        if len(words) == 0 or num_sentences == 0:
            return 0.0
        
        total_words = len(words)
        total_syllables = sum(self.count_syllables_pt(w) for w in words)
        
        asl = total_words / num_sentences
        asw = total_syllables / total_words
        
        flesch = 248.835 - (1.015 * asl) - (84.6 * asw)
        return max(0.0, min(100.0, flesch))

    def check_vides_metadata(self, meta_file):
        meta_filepath = os.path.join(self.raw_path, meta_file)
        if not os.path.exists(meta_filepath):
            return 0, 0
            
        try:
            with open(meta_filepath, encoding="utf-8") as f:
                data = json.load(f)
                
            documentos = data.get("DetalheDocumento", {}).get("documentos", {}).get("documento", [])
            if not documentos:
                return 0, 0
                
            doc = documentos[0]
            vides_container = doc.get("vides", {})
            if not vides_container or isinstance(vides_container, str):
                return 0, 0
                
            vides_list = vides_container.get("vide", [])
            if isinstance(vides_list, dict):
                vides_list = [vides_list]
                
            teve_veto = 0
            instavel = 0
            
            for vide in vides_list:
                if not isinstance(vide, dict):
                    continue
                nome = vide.get("nomeNormaPosterior", "").lower()
                coment = vide.get("comentario", "").lower()
                
                if "veto" in nome or "veto" in coment:
                    teve_veto = 1
                if any(x in coment for x in ["alteração", "revogação", "revogado"]):
                    instavel = 1
                    
            return teve_veto, instavel
        except Exception:
            return 0, 0

    def run_pipeline(self):
        if not os.path.exists(self.clean_file):
            return False
            
        df = pd.read_csv(self.clean_file)
        feature_records = []
        
        for _, row in df.iterrows():
            sentencas = json.loads(row["sentencas"])
            tokens = json.loads(row["tokens"])
            
            num_sentencas = len(sentencas)
            num_tokens = len(tokens)
            
            if num_sentencas == 0 or num_tokens == 0:
                continue
                
            texto_orig = row["texto_original"]
            words_orig = [w for w in re.findall(r"\b\w+\b", texto_orig.lower())]
            flesch = self.calculate_flesch(words_orig, num_sentencas)
            
            palavras_por_sentenca = num_tokens / num_sentencas
            
            artigos_matches = re.findall(r"\bart\.\s*\d+", texto_orig.lower())
            num_artigos = len(artigos_matches)
            sentencas_por_artigo = num_sentencas / max(1, num_artigos)
            
            ref_matches = re.findall(r"\b(art\.|lei\s*nº|§|inciso|parágrafo|alínea|caput)\b", texto_orig.lower())
            densidade_referencias = len(ref_matches) / num_tokens
            
            tokens_unicos = len(set(tokens))
            ttr = tokens_unicos / num_tokens
            
            termos_tecnicos_count = sum(1 for t in tokens if t in self.termos_juridicos)
            proporcao_termos_tecnicos = termos_tecnicos_count / num_tokens
            
            teve_veto, instavel = self.check_vides_metadata(row["meta_file"])
            
            feature_records.append({
                "id": row["id"],
                "tipo": row["tipo"],
                "numero": row["numero"],
                "ano": row["ano"],
                "data": row["data"],
                "ementa": row["ementa"],
                "flesch": flesch,
                "palavras_por_sentenca": palavras_por_sentenca,
                "sentencas_por_artigo": sentencas_por_artigo,
                "densidade_referencias": densidade_referencias,
                "ttr": ttr,
                "proporcao_termos_tecnicos": proporcao_termos_tecnicos,
                "teve_veto": teve_veto,
                "instavel": instavel
            })
            
        if not feature_records:
            return 0
            
        df_feat = pd.DataFrame(feature_records)
        
        flesch_norm = (df_feat["flesch"] - df_feat["flesch"].min()) / max(1e-5, df_feat["flesch"].max() - df_feat["flesch"].min())
        
        pps_norm = 1.0 - (df_feat["palavras_por_sentenca"] - df_feat["palavras_por_sentenca"].min()) / max(1e-5, df_feat["palavras_por_sentenca"].max() - df_feat["palavras_por_sentenca"].min())
        
        ttr_norm = (df_feat["ttr"] - df_feat["ttr"].min()) / max(1e-5, df_feat["ttr"].max() - df_feat["ttr"].min())
        
        ref_norm = 1.0 - (df_feat["densidade_referencias"] - df_feat["densidade_referencias"].min()) / max(1e-5, df_feat["densidade_referencias"].max() - df_feat["densidade_referencias"].min())
        
        tech_norm = 1.0 - (df_feat["proporcao_termos_tecnicos"] - df_feat["proporcao_termos_tecnicos"].min()) / max(1e-5, df_feat["proporcao_termos_tecnicos"].max() - df_feat["proporcao_termos_tecnicos"].min())
        
        spa_norm = 1.0 - (df_feat["sentencas_por_artigo"] - df_feat["sentencas_por_artigo"].min()) / max(1e-5, df_feat["sentencas_por_artigo"].max() - df_feat["sentencas_por_artigo"].min())
        
        score = (
            2.5 * flesch_norm +
            2.0 * pps_norm +
            2.0 * ttr_norm +
            1.5 * ref_norm +
            1.0 * tech_norm +
            1.0 * spa_norm
        )
        
        score_min = score.min()
        score_max = score.max()
        df_feat["score_qualidade"] = (score - score_min) / max(1e-5, score_max - score_min) * 10.0
        
        df_feat.to_csv(self.output_file, index=False, encoding="utf-8")
        return len(df_feat)

if __name__ == "__main__":
    extractor = LegislacaoFeatureExtractor()
    total = extractor.run_pipeline()
    print(f"Extração de features concluída. Total de leis processadas: {total}")
