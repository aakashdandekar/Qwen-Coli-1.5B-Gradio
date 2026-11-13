# Qwen-Coli-1.5B Test

Gradio streaming chat server for testing [aakashdandekar2006/Qwen-Coli-1.5B](https://huggingface.co/aakashdandekar2006/Qwen-Coli-1.5B), a language model for *Escherichia coli* codon optimization.

## Run

```bash
uv run main.py
```

The server downloads the model into `model/` automatically if not already present, then starts on `http://127.0.0.1:7860` with a Gradio share link.
