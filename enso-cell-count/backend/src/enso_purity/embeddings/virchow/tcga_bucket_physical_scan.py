import os
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

# Nome del bucket pubblico
BUCKET_NAME = "gdc-tcga-phs000178-open"

def scan_bucket():
    print(f"🕵️‍♂️ Tento di scansionare direttamente il bucket: gs://{BUCKET_NAME} ...")
    print("⚠️ NOTA: Se ricevi un errore 403 Forbidden, significa che il listing è disabilitato.")
    print("   In quel caso, dovremo fidarci per forza dell'API GDC.")
    
    try:
        # Inizializza client (tenta di usare il progetto della VM per la quota)
        # Se sei su una VM Google, prende il progetto in automatico.
        client = storage.Client()
        
        # Tentativo di listing
        # prefix=None vuol dire "dammi tutto".
        blobs_iterator = client.list_blobs(BUCKET_NAME)
        
        svs_files = []
        
        print("🚀 Scansione iniziata (potrebbe richiedere qualche minuto)...")
        
        # Iteriamo sugli oggetti
        count = 0
        for blob in tqdm(blobs_iterator, desc="Objects scanned", unit="obj"):
            count += 1
            if blob.name.endswith(".svs"):
                # La struttura solita è: UUID/NomeFile.svs
                parts = blob.name.split('/')
                
                if len(parts) >= 2:
                    uuid = parts[0]
                    filename = parts[-1]
                    size_mb = blob.size / (1024*1024)
                    
                    svs_files.append({
                        "file_id": uuid,
                        "file_name": filename,
                        "file_size_MB": round(size_mb, 2),
                        "full_path": blob.name
                    })
        
        print(f"\n✅ SCANSIONE COMPLETATA!")
        print(f"   Totale oggetti scansionati: {count}")
        print(f"   File .SVS trovati: {len(svs_files)}")
        
        if len(svs_files) > 0:
            df = pd.DataFrame(svs_files)
            output_csv = "bucket_physical_scan.csv"
            df.to_csv(output_csv, index=False)
            print(f"💾 Lista salvata in: {output_csv}")
            print(df.head())
        else:
            print("❌ Nessun file SVS trovato (strano, se il listing ha funzionato).")

    except Exception as e:
        print("\n❌ ERRORE CRITICO DURANTE LA SCANSIONE:")
        print(e)
        print("\n🔎 DIAGNOSI:")
        if "403" in str(e):
            print("Il bucket è pubblico per il DOWNLOAD, ma privato per il LISTING.")
            print("Google impedisce di vedere l'elenco file per evitare scraping massivo.")
            print("👉 DEVI USARE IL FILE 'tcga_open_svs.csv' GENERATO DALL'API (STEP PRECEDENTE).")
        else:
            print("Errore generico di connessione o configurazione gcloud.")

if __name__ == "__main__":
    scan_bucket()