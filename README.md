# DeepEval Workshop - Gemini API

## Purpose
Basic DeepEval POC demonstrating:
- Gemini as the LLM
- DeepEval Hallucination Metric
- Baseline response evaluation
- Hallucination detection

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Configuration
Create a `.env` file:
GOOGLE_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.6-flash

## Dataset
The evaluation scenarios are available in:
dataset/evaluation_data.xlsx

## Run
python -m src.main

## Report
The evaluation report is generated at:
reports/evaluation_report.xlsx

## Scenarios
TC001 - Baseline / supported response
TC002 - Hallucination demonstration