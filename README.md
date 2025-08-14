# Conversational Banking Pre-POC Discovery App

A Streamlit app to run a structured discovery for revamping an NLP chatbot to a GenAI-based **Conversational Banking** system.

## Features
- 30 **fixed** discovery questions (configurable in `questions_fixed.json`)
- **3 open-ended** questions (configurable), each spawning **5 dynamic follow-ups** via OpenAI
- All inputs saved to **MongoDB**
- Automated **maturity scoring** across 5 pillars + **Next Steps** recommendations
- One-click **report** export (Markdown)

## Quick Start
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...           # or set in your shell/profile
export MONGO_URI="mongodb+srv://..." # optional for local testing
streamlit run app.py
```

## Configuration
- Edit `config.ini` to change:
  - Open-ended prompts
  - Follow-up generation template
  - OpenAI model/temperature
  - MongoDB DB/collection names
- Edit `questions_fixed.json` to add/remove/retitle the 30 fixed questions.
- Edit `scoring_rules.json` to tune keyword heuristics, weights, thresholds, and next steps text.

## Data Model (MongoDB)
Each assessment document:
```json
{
  "org": {"name": "...", "contact": "..."},
  "answers": {
     "fixed": [{"id":"Q01","answer":"..."}, ...],
     "open": [{"prompt":"...", "answer":"...", "followups":[{"q":"...","a":"..."}]}]
  },
  "scores": {"pillars":[{"name":"...","score":12,"stage":"Developing"}],"overall":58},
  "created_at": "ISO8601"
}
```

## Notes
- If `OPENAI_API_KEY` is missing, the app will simulate follow-ups.
- If `MONGO_URI` is missing, the app will skip DB save and warn in the UI.