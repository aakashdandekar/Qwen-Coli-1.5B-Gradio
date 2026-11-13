import html
import json
from pathlib import Path
from threading import Event, Thread

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, TextIteratorStreamer

BASE_DIR = Path(__file__).resolve().parent

HOST = "127.0.0.1"
PORT = 7860
REPO_ID = "aakashdandekar2006/Qwen-Coli-1.5B"
MODEL_DIR = BASE_DIR / "model"
SYSTEM = "You are Qwen-Coli, a specialized assistant for E. coli codon optimization and biological sequence design."

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        if not (MODEL_DIR / "model.safetensors").is_file():
            print(f"Model not found in '{MODEL_DIR}'. Downloading '{REPO_ID}' from Hugging Face...")
            snapshot_download(repo_id=REPO_ID, local_dir=MODEL_DIR)
        print(f"Loading model from '{MODEL_DIR}' on {device}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        if hasattr(_model, "generation_config") and _model.generation_config is not None:
            _model.generation_config.max_length = None
    return _tokenizer, _model


def normalize_history(history):
    norm = []
    if not history:
        return norm
    for item in history[-12:]:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content")
            if content is None:
                continue
            if isinstance(content, list):
                content = "".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
            content_str = str(content).strip()
            if content_str:
                norm.append({"role": role, "content": content_str})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            u_text = str(item[0]).strip() if item[0] is not None else ""
            a_text = str(item[1]).strip() if item[1] is not None else ""
            if u_text:
                norm.append({"role": "user", "content": u_text})
            if a_text:
                norm.append({"role": "assistant", "content": a_text})
    return norm


def chat(message, history):
    tok, mdl = get_model()
    messages = [{"role": "system", "content": SYSTEM}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": message})

    try:
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
    except Exception as exc:
        yield f"Error building prompt: {exc}"
        return

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    stop_event = Event()
    stopping_criteria = StoppingCriteriaList([lambda ids, scores, **_: stop_event.is_set()])
    generation_error = []

    def generate_worker():
        try:
            mdl.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                stopping_criteria=stopping_criteria,
            )
        except Exception as exc:
            generation_error.append(str(exc))
            streamer.end()

    thread = Thread(target=generate_worker, daemon=True)
    thread.start()

    out = ""
    try:
        for piece in streamer:
            out += piece
            yield out
    finally:
        stop_event.set()
        thread.join(timeout=2.0)

    if generation_error:
        yield out + f"\n\nGeneration error: {generation_error[0]}"


EXAMPLES = [
    "Optimize this codon sequence for E. coli BL21(DE3): ATGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGAC",
    "Codon-optimize for E. coli K-12 (~50% GC): ATGAGACGAAGGCTACTACTAACTCCTCCAATAATAATACTACTATAA",
    "Optimize for E. coli and exclude BamHI/EcoRI sites: ATGGCTAGCATGACTGGTGGACAGCAAATGGGTCGCGGATCCGAATTCGAGCTCGTCGACAAGCTT",
]

HEAD_SCRIPT = """
<script>
function copyPrompt(btn, text) {
  const onSuccess = () => {
    const clip = btn.querySelector('.clip-icon');
    const check = btn.querySelector('.check-icon');
    if (clip && check) {
      clip.style.display = 'none';
      check.style.display = 'inline';
      setTimeout(() => {
        clip.style.display = 'inline';
        check.style.display = 'none';
      }, 1500);
    }
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(onSuccess).catch(() => fallbackCopy(text, onSuccess));
  } else {
    fallbackCopy(text, onSuccess);
  }
}
function fallbackCopy(text, cb) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); cb(); } catch(e) {}
  document.body.removeChild(ta);
}
</script>
"""


def build_header():
    items = []
    for ex in EXAMPLES:
        escaped_text = html.escape(ex)
        attr_text = html.escape(json.dumps(ex), quote=True)
        items.append(
            f'''<div class="example-item">
  <span class="example-text">{escaped_text}</span>
  <button class="copy-btn" onclick="copyPrompt(this, {attr_text})" title="Copy to clipboard" aria-label="Copy to clipboard">
    <svg class="clip-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>
    <svg class="check-icon" style="display:none;" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
  </button>
</div>'''
        )
    items_html = "\n".join(items)

    return f"""
<style>
.qwen-header {{
  padding: 12px 16px 6px 16px;
  display: flex;
  align-items: center;
}}
.qwen-title-row {{
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}}
.qwen-title {{
  margin: 0;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--body-text-color, inherit);
}}
.info-wrapper {{
  position: absolute;
  right: 0;
  display: inline-flex;
  align-items: center;
}}
.info-icon {{
  display: flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  cursor: pointer;
  color: #9ca3af;
  transition: color 0.2s, background-color 0.2s;
}}
.info-icon:hover, .info-wrapper:hover .info-icon {{
  color: #3b82f6;
  background-color: rgba(59, 130, 246, 0.12);
}}
.info-popover {{
  display: none;
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 6px;
  width: 440px;
  max-width: 90vw;
  background: var(--background-fill-primary, #1f2937);
  border: 1px solid var(--border-color-primary, #374151);
  border-radius: 8px;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
  padding: 12px;
  z-index: 9999;
}}
.info-popover::before {{
  content: '';
  position: absolute;
  top: -10px;
  left: 0;
  right: 0;
  height: 10px;
}}
.info-wrapper:hover .info-popover,
.info-wrapper:focus-within .info-popover {{
  display: block;
}}
.popover-heading {{
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #9ca3af;
  margin-bottom: 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--border-color-primary, #374151);
}}
.example-item {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 10px;
  border-radius: 6px;
  background: var(--background-fill-secondary, #111827);
  margin-bottom: 6px;
  font-size: 0.85rem;
  line-height: 1.4;
}}
.example-text {{
  flex: 1;
  word-break: break-word;
  color: var(--body-text-color, inherit);
}}
.copy-btn {{
  background: transparent;
  border: 1px solid var(--border-color-primary, #4b5563);
  border-radius: 4px;
  padding: 4px 6px;
  cursor: pointer;
  color: #9ca3af;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: all 0.15s;
}}
.copy-btn:hover {{
  color: #ffffff;
  border-color: #3b82f6;
  background: rgba(59, 130, 246, 0.15);
}}
</style>
<div class="qwen-header">
  <div class="qwen-title-row">
    <h1 class="qwen-title">Qwen-Coli-1.5B</h1>
    <div class="info-wrapper">
      <div class="info-icon" title="Prompt Examples" tabindex="0">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="16" x2="12" y2="12"></line>
          <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
      </div>
      <div class="info-popover">
        <div class="popover-heading">Prompt Examples</div>
        {items_html}
      </div>
    </div>
  </div>
</div>
"""


with gr.Blocks(title="Qwen-Coli-1.5B", fill_height=True) as demo:
    gr.HTML(build_header())
    gr.ChatInterface(chat, title=None, concurrency_limit=1)


if __name__ == "__main__":
    get_model()  # Pre-warm model on startup
    demo.queue().launch(
        server_name=HOST,
        server_port=PORT,
        head=HEAD_SCRIPT,
        share=True,
    )
