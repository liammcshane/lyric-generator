# Lyric Generator

Streamlit app for generating lyrics using a fine-tuned GPT-2 model.

## Model Information
- Base model: GPT-2
- Fine-tuned on: Custom lyric dataset

## Deployment Structure
```
streamlit_files/
├── model/
│   ├── config.json
│   ├── model.safetensors (or pytorch_model.bin)
│   └── [tokenizer files if available]
├── requirements.txt
└── README.md
```

## Next Steps
1. Download the `app.py` file for the Streamlit interface
2. Upload all files to GitHub repository
3. Deploy on Streamlit Cloud

## Usage
The app will use the original GPT-2 tokenizer with your fine-tuned model weights.
