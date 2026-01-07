# PREDICTING NEXT WORD USING LSTM

*Neural Language Modeling with LSTM for Next-Word Prediction*

![python](https://img.shields.io/badge/python-3.8%2B-blue)
![nlp](https://img.shields.io/badge/nlp-language_modeling-green)
![dl](https://img.shields.io/badge/deep_learning-lstm-success)

---

## Overview

This project implements a **next-word prediction system** using a **Long Short-Term Memory (LSTM)** neural network.

The model learns sequential patterns in text data and predicts the most likely next word given a sequence of previous words.
The project focuses on **core NLP fundamentals**: text preprocessing, sequence modeling, and neural language modeling.

---

## Why This Project Matters (Recruiter Focus)

* Demonstrates understanding of **sequence modeling**
* Hands-on implementation of **LSTM for NLP**
* Covers the full NLP pipeline (text → tokens → sequences → predictions)
* Shows ability to work beyond tabular ML into **deep learning & NLP**

This project builds foundational skills that transfer to chatbots, autocomplete systems, and text generation models.

---

## Problem Statement

Given a sequence of words, predict the **next most probable word** based on learned language patterns.

This is a classic **language modeling** problem and a precursor to more advanced NLP architectures such as Transformers.

---

## NLP Workflow

1. **Text Ingestion**

   * Load raw text corpus

2. **Text Preprocessing**

   * Tokenization
   * Vocabulary creation
   * Integer encoding

3. **Sequence Generation**

   * Sliding window sequences
   * Last word used as prediction target

4. **Model Architecture**

   * Embedding layer
   * LSTM layer(s)
   * Dense + Softmax output layer

5. **Training**

   * Train model on word sequences
   * Optimize categorical cross-entropy loss

6. **Inference**

   * Predict next word for a given input phrase

---

## Project Structure

```
Predicting-Next-Word-Using-LSTM-/
│
├── notebooks/            # Model development & training
├── data/                 # Text corpus
├── model/                # Saved LSTM model
│
├── requirements.txt
└── README.md
```

*(Exact file names may vary depending on implementation style.)*

---

## Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **NLP:** Tokenization, sequence padding
* **Data Handling:** NumPy
* **Development:** Jupyter Notebook

---

## Run the Project

```bash
git clone https://github.com/Muheet-Mehraj/Predicting-Next-Word-Using-LSTM-.git
cd Predicting-Next-Word-Using-LSTM-
pip install -r requirements.txt
```

Run the notebook to:

* preprocess text
* train the LSTM model
* generate next-word predictions

---

## Model Notes

* **Model Type:** LSTM-based Neural Language Model
* **Prediction Type:** Multiclass word prediction (Softmax)
* **Training Objective:** Learn contextual word dependencies

> The project emphasizes **correct NLP pipeline design** rather than benchmarking against large pretrained models.

---

## Engineering Highlights (Recruiter-Friendly)

* Sequence modeling with LSTM
* Proper text tokenization and padding
* Vocabulary management
* Neural network training for NLP tasks
* Reusable inference logic
* Clear separation of preprocessing and modeling

---

## Limitations

* Performance depends on corpus size and quality
* Vocabulary is fixed after training
* Not comparable to transformer-based language models
* No beam search or sampling strategies implemented

---

## Future Improvements

* Pretrained word embeddings (GloVe / FastText)
* Beam search or temperature-based sampling
* REST API or web UI for predictions
* Migration to Transformer-based models

---

## Author

**Muheet Mehraj**
B.Tech CSE 
GitHub: [https://github.com/Muheet-Mehraj](https://github.com/Muheet-Mehraj)

