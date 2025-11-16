# Fake News Detection Using NLP & Machine Learning

**Repository**: Fake-News-Detection
**Author**: Koustab Dutta
**Contact**: `koustab@example.com` (replace with your email)
**License**: MIT

A reproducible project that trains a text classifier to detect **fake vs real news** using NLP techniques and Transformer-based models. The project includes model training code, evaluation, and a simple web demo (text + URL analysis) with UI screenshots included.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Highlights / Features](#highlights--features)
3. [Repository structure](#repository-structure)
4. [Dataset](#dataset)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Quickstart — train & evaluate](#quickstart--train--evaluate)
8. [Inference / Demo](#inference--demo)
9. [Reproduce results (notes)](#reproduce-results-notes)
10. [Screenshots / Demo images](#screenshots--demo-images)
11. [Contributing](#contributing)
12. [Citation / Acknowledgements](#citation--acknowledgements)
13. [License](#license)

---

# Project overview

This project implements a pipeline to detect fake news using NLP techniques. The main approach tested was fine-tuning a DistilBERT (transformer) classifier for binary classification (`real` vs `fake`). Baseline classical models (Logistic Regression / SVM / Naive Bayes) were also evaluated for comparison.

The notebook `Final_Fake_News_Detection.ipynb` contains data-preparation, model training, evaluation code and demonstration examples. A simple web UI demonstrates text and URL analysis with confidence scores.

---

# Highlights / Features

* End-to-end pipeline: data cleaning → feature extraction → model training → evaluation → demo
* Transformer-based final model: **DistilBERT** (fine-tuned for binary classification)
* Training stack: `transformers`, `torch` (GPU acceleration if available), `scikit-learn`
* Evaluations: accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC
* Demo UI for text & URL analysis (screenshots included)
* Reproducible scripts + notebook for step-by-step reproduction

---

# Repository structure

```
.
├── README.md
├── requirements.txt
├── Final_Fake_News_Detection.ipynb
├── data/
│   └── (place dataset CSV/JSON files here)
├── src/
│   ├── train.py                # training script (HuggingFace transformers)
│   ├── evaluate.py             # evaluation utilities and metric reporting
│   ├── infer.py                # single-text & batch inference
│   └── web_app.py              # demo Flask/Streamlit app for text & URL analysis
├── models/
│   └── distilbert_finetuned/   # saved model (transformers `save_pretrained`)
├── assets/
│   ├── Fake.png
│   ├── Real.png
│   ├── url_fake.png
│   └── url_real.png
└── docs/
    └── presentation/           # pptx / slides for project
```

---

# Dataset

Place your dataset in the `data/` folder. Typical dataset columns used:

* `title` (optional)
* `text` (full article or headline)
* `label` (`real` / `fake` or 0 / 1)

> **Note:** The notebook expects a labeled dataset (common benchmarks include Kaggle Fake News datasets). If you used a custom dataset, keep the same column names or update the data-loading cell in the notebook.

---

# Requirements

Create a conda environment or use pip. Example `requirements.txt` (add versions as needed):

```
torch
transformers
datasets
scikit-learn
pandas
numpy
tqdm
flask           # or streamlit if using Streamlit for demo
newspaper3k     # for scraping article text from URL (optional)
beautifulsoup4
requests
matplotlib
```

Install:

```bash
pip install -r requirements.txt
# or using conda
conda create -n fake-news python=3.10
conda activate fake-news
pip install -r requirements.txt
```

---

# Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/<your-username>/fake-news-detection.git
   cd fake-news-detection
   ```
2. Install requirements (see previous section).
3. Put your dataset in `data/` (e.g., `data/news.csv`).
4. (Optional) Create a `models/` directory to save trained models.

---

# Quickstart — train & evaluate

### Training (example using Hugging Face / Transformers)

Below is a minimal example; adapt to the notebook code.

```python
# src/train.py (simplified example)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import datasets
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load dataset (example: CSV with 'text' and 'label' columns)
data = datasets.load_dataset('csv', data_files={'train':'data/train.csv', 'test':'data/test.csv'})

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def preprocess(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)

data = data.map(preprocess, batched=True)
data = data.rename_column("label", "labels")
data.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./models/distilbert_finetuned',
    num_train_epochs=1,            # notebook used epochs=1 (adjust as needed)
    per_device_train_batch_size=16,# batch_size=16
    per_device_eval_batch_size=32,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=100,
    learning_rate=2e-5,            # typical for fine-tuning with AdamW
    fp16=torch.cuda.is_available()
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data['train'],
    eval_dataset=data['test'],
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./models/distilbert_finetuned")
```

### Evaluate

```bash
python src/evaluate.py --model ./models/distilbert_finetuned --test data/test.csv
```

`evaluate.py` should compute and print accuracy, precision, recall, F1, and optionally save a confusion matrix and ROC plot.

---

# Inference / Demo

A simple inference snippet using the saved model:

```python
# src/infer.py
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np

model = DistilBertForSequenceClassification.from_pretrained("./models/distilbert_finetuned")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model.eval()

def predict_text(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt', max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    label = "real" if pred == 0 else "fake"   # adapt to your label mapping
    return {"label": label, "confidence": confidence, "probs": probs.tolist()}

print(predict_text("Local Scientists Discover New, Highly Durable Alloy That Bends Light"))
```

### Demo web app

The notebook includes code for a text and URL analysis UI. A simple Flask route example:

```python
# src/web_app.py (very minimal)
from flask import Flask, request, render_template, jsonify
from infer import predict_text
from newspaper import Article

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # basic UI with text box and URL box

@app.route('/predict_text', methods=['POST'])
def predict_text_route():
    text = request.form['text']
    result = predict_text(text)
    return jsonify(result)

@app.route('/predict_url', methods=['POST'])
def predict_url_route():
    url = request.form['url']
    article = Article(url)
    article.download(); article.parse()
    text = article.title + "\n\n" + article.text
    result = predict_text(text)
    result['title'] = article.title
    result['source'] = url
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

# Reproduce results — notes & tips

* The notebook used:

  * Model: **DistilBERT** (Hugging Face `DistilBertForSequenceClassification`) — transformer fine-tune
  * Optimizer: **AdamW**
  * Batch size: **16**
  * Epochs: **1** (this was used in the notebook; for better performance use more epochs as needed)
  * Metrics: accuracy, precision, recall, F1-score (computed with `sklearn.metrics`)
* Run experiments on GPU (NVIDIA RTX recommended) for speed.
* Fix random seeds for reproducibility (set seeds for `numpy`, `torch`, and Transformers `TrainingArguments` where applicable).
* If your dataset is small, consider k-fold cross-validation or stratified splits for stable evaluation.

---

# Screenshots / Demo images

Demo screenshots (already included in repository under `assets/`):

* `assets/Fake.png` — UI showing a "Fake News" detection example
* `assets/Real.png` — UI showing a "Real News" detection example
* `assets/url_fake.png` — URL analysis returning "Fake"
* `assets/url_real.png` — URL analysis returning "Real"

Use these in your README or presentation to showcase the demo.

---

# Contributing

Contributions are welcome! Typical workflow:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push branch and open a pull request

Please add tests or a small example notebook showing the change when appropriate.

---

# Citation / Acknowledgements

If you use this work in a publication or project, please cite the repository and any datasets used (e.g., the Kaggle dataset if used). Example citation (adapt to final paper):

> Dutta, K. (2025). *Fake News Detection Using NLP & Machine Learning*. GitHub repository. [https://github.com/](https://github.com/)<your-username>/fake-news-detection

---

# License

This project is released under the **MIT License** — see the `LICENSE` file for details.

---
