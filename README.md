# MedGemma 1.5 4B — Dermatoscopie Edge & Eval Harness

Architecture hybride (**code Python versionné en local + banc GPU Google Colab**) combinant :

1. **Contrôle qualité pré-analytique 100 % local (CPU)** : bloque avant inférence les clichés flous, sous/surexposés, saturés de reflets spéculaires, envahis par les poils ou mal cadrés (réponse directe aux refus en télédermatologie et au *Shadow AI* hors RGPD).
2. **Sortie sémiologique structurée (Pydantic v2)** : description des critères dermatoscopiques observables (asymétrie, réseau pigmentaire, globules, stries, voile bleu-blanc, patron vasculaire, régression, couleurs), calibration et **abstention obligatoire** en cas de doute ou d'image insuffisante.
3. **Ingestion ISIC Archive sans fuite de données** :
   - Traçabilité par image de `copyright_license` avec mode `--commercial-only` (`CC-0`, `CC-BY`, `CC-BY-4.0`).
   - Split groupé par `patient_id` / `lesion_id` / `isic_id`.
   - **Politique anti-leakage MedGemma 1.5** : `google/medgemma-1.5-4b-it` ayant intégré ~40 269 exemples d'ISIC 2017/2018 CC-0 au pré-entraînement, les collections ISIC 2017, ISIC 2018 et HAM10000 (`[60, 62, 63, 64, 66, 67, 69, 71, 73, 212]`) sont systématiquement exclues du jeu de test verrouillé (`test.jsonl`).
4. **Eval Harness verrouillé** : mesure la validité JSON, la sensibilité directe et la sensibilité de triage sûr (`TP + abstentions`), la spécificité, l'ECE et la régression `BF16` vs `QLoRA` vs `GGUF`.

---

## Positionnement réglementaire & Licence (UE / HAI-DEF)

- **Usage Research Use Only (RUO) & Hors-MDR :** ce dépôt ne constitue pas un dispositif médical (MDR Classe IIa) et n'émet aucun diagnostic clinique autonome. Les poids `google/medgemma-1.5-4b-it` sont soumis aux conditions **Health AI Developer Foundations (HAI-DEF)** de Google.
- **Anti-Shadow-AI :** toute exécution locale de la brique pre-flight et de structuration reste hors cloud sans fuite de données patient.

---

## Quickstart local (0 Compute Unit Colab)

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
.venv\Scripts\pytest.exe -v
.venv\Scripts\ruff.exe check src scripts tests
```

### Construction du dataset (exemple avec audit de licence)

```powershell
.venv\Scripts\python.exe scripts/build_dataset.py --commercial-only --max-items 500 --output-dir data/isic_v1
```

---

## Workflow Colab (L4 pour baseline, A100 pour QLoRA)

1. Accepter la licence HAI-DEF sur `google/medgemma-1.5-4b-it` et enregistrer `HF_TOKEN` dans les **Colab Secrets** (jamais dans le repo).
2. Ouvrir `notebooks/colab_runner.ipynb` sur Google Colab (choisir **L4** pour la baseline zéro-shot, **A100 40 Go** pour QLoRA `bfloat16` — éviter T4 qui ne supporte pas `bfloat16` nativement).
3. Monter Google Drive pour persister dataset et checkpoints (`save_steps: 50`), exécuter `scripts/run_baseline.py` puis `scripts/train_qlora.py`, et déconnecter le runtime dès la fin du run.
