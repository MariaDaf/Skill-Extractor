# Skill-Extractor

This project implements an NLP pipeline for extracting professional skills from German-language job advertisements.

The goal is to turn unstructured job-ad text into a clean list of skill phrases.

## Project Idea

Job advertisements contain useful information such as skills, qualifications, experience, benefits, and application details. However, this information is usually written as free text and is not directly structured.

Our pipeline works in two main stages:

```text
Raw job advertisement
        ↓
BERT zone classifier
        ↓
Extract "Fähigkeiten und Inhalte" section
        ↓
Gemini skill extraction
        ↓
Clean list of professional skills

BERT is used to identify the relevant zones in the job ad. The most important zone is Fähigkeiten und Inhalte, because it contains the skill-related text. This extracted text is then passed to Gemini, which returns skills as short phrases.

## Repository Structure
Skill-Extractor/
│
├── data/
│   ├── train_dataset.pt
│   ├── test_dataset.pt
│   ├── crawled_jobs.csv
│   ├── crawled_jobs.json
│   ├── extracted_skills.csv
│   ├── extracted_skills.json
│   └── raw_txt/
│
├── model/
│   └── id2label.json
│
├── scripts/
│   ├── preprocessing.py
│   ├── train_bert_zone_classifier.py
│   ├── task4_crawler.py
│   └── task5_pipeline.py
│
├── Part1_Documentation.docx
├── Part2_Documentation_extension.docx
├── Skill_Extraction_Project.pdf
├── Team_Agreement_Advanced_Gen_AI.pdf
├── .gitignore
└── README.md

------------------------------------------------------------------------------

## Part 1 — Data Preparation and Preprocessing

Part 1 prepares the annotated job-ad dataset for BERT training.

The preprocessing script converts character-level annotations into token-level labels. This is necessary because BERT does not work directly with full text spans, but with tokens.

Main outputs:

train_dataset.pt
test_dataset.pt
model/id2label.json

The datasets contain token IDs, attention masks, and token-level labels. Padding labels are stored as -100 so that they are ignored during training.

## Part 2 — BERT Training and Evaluation

Part 2 trains a BERT token-classification model to predict the zone of each token in a job advertisement.

Main script:

scripts/train_bert_zone_classifier.py

The model is based on:

bert-base-multilingual-cased

The training pipeline includes:

loading the preprocessed datasets
loading the label mapping
training BertForTokenClassification
using class weights for imbalanced labels
using a learning-rate scheduler
logging metrics with TensorBoard
saving the best model checkpoint
evaluating the model with precision, recall, and F1-score

The BERT model successfully trains and evaluates end-to-end. However, the final span-level F1-score is still relatively low, which means that exact zone-boundary prediction remains challenging.

The trained model checkpoint is too large for GitHub and is therefore shared separately.

Required checkpoint for inference:

best_student_model.pt

Required label mapping:

model/id2label.json

##Part 3 — Web Crawling and Gemini Skill Extraction

Part 3 applies the full pipeline to fresh real-world job advertisements.

It includes:

crawling German-language job ads from the SBB career portal
cleaning the advertisement text
applying the trained BERT model to identify the Fähigkeiten und Inhalte zone
sending this zone text to Gemini
parsing Gemini’s response into structured skill phrases
saving the results as JSON and CSV files

Main outputs:

data/crawled_jobs.csv
data/crawled_jobs.json
data/extracted_skills.csv
data/extracted_skills.json
Final Product

The final product is a Python-based NLP pipeline that takes a job advertisement as input and returns a structured list of professional skills.

Example input:

We are looking for a Data Analyst with experience in Python, SQL, Power BI and data visualization.

Example output:

[
  "Python programming",
  "SQL databases",
  "Power BI reporting",
  "data visualization"
]

## Limitations

The pipeline works end-to-end, but the main limitation is the BERT zone classifier. If BERT extracts a clean Fähigkeiten und Inhalte section, Gemini can produce useful skill phrases. If BERT produces fragmented or noisy text, the quality of the final skill list decreases.

The model is therefore a working baseline, but not yet production-ready.

Potential Future Work

Future work should focus on improving the BERT zone classifier. Possible improvements include:

better handling of class imbalance
capped class weights
manual token-level error analysis
improved post-processing of predicted zones
separate evaluation of the Fähigkeiten und Inhalte zone
a dedicated inference script for using the trained model without retraining

A useful next step would be to create a simple prediction script that loads the trained BERT checkpoint, takes a new job ad as input, predicts the zones, and returns only the extracted Fähigkeiten und Inhalte section for Gemini.
