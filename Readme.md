# Language Model Collaboration for Relation Extraction from Classical Chinese Historical Documents

This repository contains the code and data for the paper "Language model collaboration for relation extraction from classical Chinese historical documents". It includes an implementation of the SpERT small model and an SLM+LLM collaboration framework, together with evaluation scripts.

## Data
- Data is stored in the `chisre/` directory.
- `test.json` and `chisre_test.json` contain the same records but use different formats. Choose the file required by each script.

## Code layout
- `span/` — SpERT small-model implementation (see: https://arxiv.org/abs/1909.07755).
- `SLCoLM/` — Implementation and examples for the SLM+LLM collaboration framework.

Main scripts:
- `python model_collaboration.py`  
    Generate relation triplets based on SpERT predictions, relation type definitions, and example-driven outputs. 
- `python eval.py`  
    Evaluate generated outputs. Verify the evaluation script input path and expected format before running.

## Usage recommendations
- Use Python 3.8+ and install project dependencies (if a `requirements.txt` file is provided).
- Before running, confirm paths for data, model weights, and configuration files. Adjust path variables inside scripts if needed.
- If you encounter format or path issues, check the top of each script for configurable variables or search the repository for `config` / `paths` settings.

## Citation
If you use this work, please cite:

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
