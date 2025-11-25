
# 📜 Language Model Collaboration for Relation Extraction from Classical Chinese Historical Documents

This repository contains the code and data accompanying the paper **“Language Model Collaboration for Relation Extraction from Classical Chinese Historical Documents.”**
It provides implementations for the **SpERT small model** and a **Small-Language-Model + Large-Language-Model (SLM+LLM) collaboration framework**, along with evaluation scripts.

---

## 📂 Data

* All datasets are located in the `chisre/` directory.
* `test.json` and `chisre_test.json` contain the **same records** but in **different formats**.
  ➤ Use the file required by each specific script.

---

## 🧭 Repository Structure

### 🔧 Code Layout

* `./code/span/` — SpERT small-model implementation
  （Reference: [https://arxiv.org/abs/1909.07755）](https://arxiv.org/abs/1909.07755）)
* `./code/SLCoLM/` — SLM+LLM collaboration framework (code + examples)

### ▶️ Main Scripts in SLCoLM

* **`python model_collaboration.py`**
  🤝 Generates relation triplets using SpERT predictions, relation schemas, and example-driven generation.
* **`python eval.py`**
  📝 Evaluates the generated outputs.
  Make sure the input file path and expected format match what the script requires.

---

## 💡 Usage Recommendations

* Use **Python ≥ 3.8**.
* Install dependencies via `requirements.txt` (if provided).
* Before running scripts, double-check:

  * 📁 Data paths
  * 🧩 Model weight paths
  * ⚙️ Config files
* If you encounter errors, inspect the top of the script for configurable variables or search for `config` or `paths` in the repo.

---

## 📖 Citation

If you use this repository in your research, please cite:

```
@article{Tang_Wang_Wang_2026,
  title={Language model collaboration for relation extraction from classical Chinese historical documents},
  volume={63},
  ISSN={0306-4573},
  DOI={10.1016/j.ipm.2025.104286},
  number={1},
  journal={Information Processing & Management},
  author={Tang, Xuemei and Wang, Linxu and Wang, Jun},
  year={2026},
  month=jan,
  pages={104286}
}
```

---

## 📬 Contact

If you have any questions, feel free to:

* 🐞 Open an issue
* ✉️ Email me at **[xuemeitang00@gmail.com](mailto:xuemeitang00@gmail.com)**
