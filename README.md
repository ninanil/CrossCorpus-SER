# Feature and Classifier-Level Domain Adaptation in DistilHuBERT for Cross-Corpus Speech Emotion Recognition

This repository provides the implementation of feature-level and classifier-level domain adaptation techniques using the DistilHuBERT model for **Cross-Corpus Speech Emotion Recognition (CCSER)**. Our goal is to improve emotion recognition performance across mismatched training and test corpora, using both lightweight and progressively fine-tuned strategies.

---

## 📁 Project Structure

- `finetune distilHuBERT.ipynb`: Implements FDA methods 2–4 by fine-tuning various components (CNN encoder or transformer blocks) of the DistilHuBERT model.
- `siamcnn.ipynb`: Implements FDA method 1 using a Siamese CNN architecture with contrastive loss.

---

## 📘 Dataset Preparation

We use three public datasets:
- **IEMOCAP**
- **ShEMO**
- **EMODB**

All audio files must be:
- Downsampled to **16 kHz**
- Padded or trimmed to **7 seconds** (112,000 samples)

### 🎯 Emotion Labels
Only four shared emotion classes are used across all datasets:
- `neutral`, `happy`, `angry`, `sad`

---

## 🗂️ Siamese Dataset Format

For contrastive training (FDA-1), a CSV file (`pairs.csv`) is used to define paired utterances. Example:

| session | imocap       | emodb       | contrastive_label |
|---------|--------------|-------------|-------------------|
| 10      | utt001.wav   | emo002.wav  | 1                 |
| 11      | utt003.wav   | emo010.wav  | 0                 |

- `contrastive_label = 1` means same emotion
- `contrastive_label = 0` means different emotion

> Audio files are preprocessed to **7-second 16kHz waveforms**, either trimmed or padded by repetition.

---

## ⚙️ W&B Hyperparameter Sweep (for FDA-1)

We use **Weights & Biases (W&B)** to manage hyperparameter sweeps. Below is an example configuration:

```python
sweep_config = {
    'method': 'grid',
    'parameters': {
        'fold': {'values': [3, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
        'epochs': {'value': 50},
        'batch_size': {'value': 8},
        'learning_rate': {'value': 3e-5},
        'weight_decay': {'value': 9e-3},
        'loss': {'value': 'contrastive'},
        'optimizer': {'value': 'AdamW'},
        'project_name': {'value': 'siameseNet'},
        'input_size': {'value': (1, 112000)},
        'sample_rate': {'value': 16000},
        'sessionListDataset': {'value': [9, 10, 11, 12, 13, 14, 15, 16]},
        'dataset': {'value': 'IEMOCAP_EMODB'},
        'source_dataset_path': {'value': '<path_to_iemocap>'},
        'target_dataset_path': {'value': '<path_to_emodb>'},
        'pair_dataset_path': {'value': '<path_to_pair_csv>'},
        'source_x_name': {'value': 'imocap'},
        'target_x_name': {'value': 'emodb'},
        'pair_name': {'value': 'pairs.csv'},
        'class_number': {'value': 4},
        'class_names': {'value': ['neutral', 'happy', 'angry', 'sad']},
        'valid_labels': {'value': ['0', '1', '2', '3']},
        'distance': {'value': 'contrastiveLoss-mmd'},
        'margin': {'value': 2},
        'sigma': {'value': 2},
        'kernel': {'value': 'rbf'},
        'freeze': {'value': True},
        'initial_weights_library': {'value': 'distilhubert'}
    }
}
```

### 🧪 Run the Sweep
```bash
wandb login
wandb sweep sweep_config.yaml
wandb agent <your-entity>/<project-name>/<sweep-id>
```

---

## 📦 Requirements
- Python >= 3.8
- PyTorch >= 1.10
- torchaudio
- pandas
- wandb
