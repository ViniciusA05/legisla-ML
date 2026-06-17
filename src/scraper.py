import requests
import pandas as pd
import time
import os
import json

class LegislacaoScraper:
    def __init__(self, base_path="data"):
        self.base_url = "https://legis.senado.leg.br/dadosabertos/legislacao/lista.json"
        self.detail_url = "https://legis.senado.leg.br/dadosabertos/legislacao/"
        self.raw_path = os.path.join(base_path, "raw")
        self.index_file = os.path.join(self.raw_path, "index.csv")
        
        os.makedirs(self.raw_path, exist_ok=True)

    def fetch_laws_by_year(self, year, sigla="LEI"):
        """Busca a lista de normas por ano."""
        params = {"sigla": sigla, "ano": year}
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            # Navega na estrutura identificada: ListaDocumento -> documentos -> documento
            return data.get('ListaDocumento', {}).get('documentos', {}).get('documento', [])
        except Exception as e:
            print(f"Erro ao buscar lista de {year}: {e}")
            return []

    def fetch_law_content(self, law_id):
        """Busca o detalhe de uma norma via ID do Senado."""
        url = f"{self.detail_url}{law_id}.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Erro ao buscar conteúdo do ID {law_id}: {e}")
            return None

    def save_json(self, data, filename):
        filepath = os.path.join(self.raw_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    scraper = LegislacaoScraper()
    year = 2024
    laws = scraper.fetch_laws_by_year(year)
    
    if laws:
        leis_puras = [l for l in laws if l.get('tipo') in ['LEI-n', 'LCP']]
        print(f"Encontradas {len(leis_puras)} Leis em {year}.")
        
        if leis_puras:
            exemplo = leis_puras[0]
            law_id = exemplo.get('id')
            print(f"Buscando detalhes da lei: {exemplo.get('normaNome')} (ID: {law_id})")
            
            details = scraper.fetch_law_content(law_id)
            if details:
                print("Detalhes obtidos com sucesso!")
                scraper.save_json(details, f"details_{law_id}.json")
            else:
                print("Falha ao obter detalhes.")
