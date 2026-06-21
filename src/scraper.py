import requests
import pandas as pd
import time
import os
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class LegislacaoScraper:
    def __init__(self, base_path="data"):
        self.base_url = "https://legis.senado.leg.br/dadosabertos/legislacao/lista.json"
        self.detail_url = "https://legis.senado.leg.br/dadosabertos/legislacao/"
        self.norma_web_url = "https://legis.senado.leg.br/norma/"
        self.raw_path = os.path.join(base_path, "raw")
        self.index_file = os.path.join(self.raw_path, "index.csv")
        os.makedirs(self.raw_path, exist_ok=True)

    def fetch_laws_by_year(self, year, sigla="LEI"):
        params = {"sigla": sigla, "ano": year}
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get("ListaDocumento", {}).get("documentos", {}).get("documento", [])
        except Exception:
            return []

    def fetch_law_metadata(self, law_id):
        url = f"{self.detail_url}{law_id}.json"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def scrape_sigen_url(self, law_id):
        url = f"{self.norma_web_url}{law_id}"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.text.strip()
                if "publicacao/" in href and "Sigen" in text:
                    return urljoin(self.norma_web_url, href)
            return None
        except Exception:
            return None

    def fetch_sigen_text(self, sigen_url):
        try:
            response = requests.get(sigen_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content_div = soup.find(id="conteudoPrincipal")
            if not content_div:
                content_div = soup.find(class_="tela")
            if content_div:
                lines = [line.strip() for line in content_div.text.split("\n")]
                return "\n".join([line for line in lines if line])
            return None
        except Exception:
            return None

    def run_pipeline(self, start_year=2020, end_year=2026, limit_per_year=30):
        records = []
        for year in range(start_year, end_year + 1):
            for sigla in ["LEI", "LCP"]:
                laws = self.fetch_laws_by_year(year, sigla)
                if not laws:
                    continue
                
                count = 0
                for law in laws:
                    if count >= limit_per_year:
                        break
                    
                    if law.get("tipo") not in ["LEI-n", "LCP"]:
                        continue
                    
                    law_id = law.get("id")
                    if not law_id:
                        continue
                    
                    metadata = self.fetch_law_metadata(law_id)
                    if not metadata:
                        continue
                    
                    sigen_url = self.scrape_sigen_url(law_id)
                    if not sigen_url:
                        continue
                    
                    text = self.fetch_sigen_text(sigen_url)
                    if not text:
                        continue
                    
                    meta_filename = f"{law_id}_meta.json"
                    text_filename = f"{law_id}_text.txt"
                    
                    with open(os.path.join(self.raw_path, meta_filename), "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=4)
                        
                    with open(os.path.join(self.raw_path, text_filename), "w", encoding="utf-8") as f:
                        f.write(text)
                        
                    records.append({
                        "id": law_id,
                        "tipo": law.get("tipo"),
                        "numero": law.get("numero"),
                        "ano": year,
                        "data": law.get("dataassinatura"),
                        "ementa": law.get("ementa", ""),
                        "meta_file": meta_filename,
                        "text_file": text_filename
                    })
                    count += 1
                    time.sleep(1)
                    
        if records:
            df = pd.DataFrame(records)
            df.to_csv(self.index_file, index=False, encoding="utf-8")
            return len(records)
        return 0

if __name__ == "__main__":
    scraper = LegislacaoScraper()
    total = scraper.run_pipeline(start_year=2020, end_year=2026, limit_per_year=30)
    print(f"Coleta concluída. Total de leis baixadas: {total}")
