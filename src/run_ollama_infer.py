# -*- coding: utf-8 -*-   
import argparse, subprocess, json, yaml, pandas as pd
from pathlib import Path
from datetime import datetime




def run_ollama(model, prompt, text, options):
    """
    Fonction qui exécute un modèle Ollama avec un prompt et un texte donné.
    - Construit le prompt complet
    - Lance la commande ollama
    - Gère les erreurs avec un fallback (stdin)
    """
    # Construit le prompt complet à envoyer au modèle
    full_prompt = f"{prompt}\n\nTEXTE:\n{text}\n\nRéponse:"


    # 1ère méthode : passer le prompt comme argument à la commande Ollama
    cmd = ["ollama", "run", model, full_prompt]
    res = subprocess.run(cmd, text=True, capture_output=True)


    if res.returncode != 0:
        # Si erreur, 2ème méthode : envoyer le prompt via stdin
        res = subprocess.run(
            ["ollama", "run", model],
            input=full_prompt,
            text=True,
            capture_output=True
        )
        # Si encore erreur : lever une exception
        if res.returncode != 0:
            raise RuntimeError(f"Ollama error: {res.stderr.strip()}")


    # Retourner la sortie texte du modèle (réponse brute)
    return res.stdout.strip()




def main(cfg_path):
    """
    Fonction principale qui :
    - Charge la configuration YAML
    - Lit les textes d’entrée
    - Appelle Ollama pour chaque texte
    - Sauvegarde les résultats (CSV) et les métadonnées (JSON)
    """


    # Charger la configuration YAML (modèle, options, chemins fichiers)
    cfg = yaml.safe_load(open(cfg_path, "r"))
    model = cfg["model"]             # Nom du modèle (ex: llama3.1:8b)
    opt = cfg.get("options", {})     # Options du modèle (température, etc.)
    io = cfg["io"]                   # Chemins des fichiers (input, prompt, output)


    # Lire le fichier de prompt
    prompt = Path(io["prompt_path"]).read_text(encoding="utf-8")
    # Définir le chemin d’entrée (texte ou CSV)
    inp = Path(io["input_path"])
    # Créer le répertoire de sortie s’il n’existe pas
    out_dir = Path(io["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)


    # Générer un horodatage unique pour les fichiers de sortie
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_in = inp.stem  # Nom de base du fichier d’entrée (sans extension)
    cfg_name = Path(cfg_path).stem   # Nom du fichier YAML de config
    run_name = f"{ts}__model-{model.replace(':','_')}__input-{base_in}__cfg-{cfg_name}"


    # Charger les données selon le type de fichier
    if inp.suffix.lower() == ".csv":
        df = pd.read_csv(inp)  # lecture CSV
        texts = df[io["input_text_col"]].astype(str).tolist()  # colonne contenant le texte
    elif inp.suffix.lower() == ".txt":
        # Lecture ligne par ligne pour un .txt
        texts = [l.strip() for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame({"text": texts})  # convertir en DataFrame pour homogénéité
    else:
        # Si ce n’est pas CSV ou TXT, erreur
        raise ValueError("input_path must be .csv with a text column or .txt (one sample per line).")


    # Listes pour stocker les résultats (prédictions et réponses brutes)
    preds, raws = [], []


    # Boucle sur chaque texte
    for t in texts:
        out = run_ollama(model, prompt, t, opt)  # appel du modèle
        raws.append(out)                         # stocker la réponse brute


        # Déterminer le label (OUI, NON ou AMBIGU)
        label = "AMBIGU"  # valeur par défaut
        for k in ["OUI", "NON", "AMBIGU"]:
            # Vérifier si la sortie contient explicitement le label attendu
            if f"LABEL: {k}" in out.upper() or out.strip().upper().startswith(k):
                label = k
                break
        preds.append(label)


    # Créer un DataFrame de sortie avec les prédictions
    df_out = df.copy()
    df_out["prediction"] = preds
    df_out["raw"] = raws


    # Sauvegarder les résultats dans un fichier CSV
    csv_path = out_dir / f"{run_name}.csv"
    df_out.to_csv(csv_path, index=False, encoding="utf-8")


    # Créer et sauvegarder les métadonnées (JSON)
    meta = {
        "timestamp": ts,
        "model": model,
        "options": opt,
        "input_path": str(inp),
        "input_rows": len(df_out),
        "prompt_path": io["prompt_path"],
        "task": cfg.get("meta", {}).get("task", ""),
        "dataset_name": cfg.get("meta", {}).get("dataset_name", ""),
        "output_csv": str(csv_path),
    }
    json_path = out_dir / f"{run_name}__meta.json"
    json.dump(meta, open(json_path, "w"), ensure_ascii=False, indent=2)


    # Afficher les chemins de sortie
    print(f"[OK] Résultats: {csv_path}")
    print(f"[OK] Métadonnées: {json_path}")




if __name__ == "__main__":
    # Parser l’argument --config pour donner le chemin du fichier YAML
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Chemin du fichier YAML de config LLM")
    args = ap.parse_args()


    # Exécuter la fonction principale avec la config donnée
    main(args.config)
