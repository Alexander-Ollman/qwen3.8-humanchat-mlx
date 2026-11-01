# Qwen3.8 Humanchat MLX

**TL;DR:** This repo serves [Qwen3.8-27B-Humanlike-Chat 2.0](https://huggingface.co/LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-2.0) on an Apple Silicon Mac through Ollama's MLX runner. It adds a local chat page with a system prompt.

No public MLX build of this model existed on 2026-10-31, so this repo builds one from the upstream BF16 safetensors. The repo holds no model weights.

## Quick start

You need Ollama 0.40.13 or later, the `hf` CLI, and Python 3.12. The download is 55.59 GB, and the Ollama build needs about 21 GB more. Run these steps from the repo root.

```bash
hf download LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-2.0 --local-dir models/bf16
(cd ollama && ollama create humanlike-2.0-nvfp4 -q nvfp4 -f Modelfile)
python3 chat/server.py
```

Open http://127.0.0.1:8091 to chat. The system prompt saves as you type. "Clear conversation" removes the messages and keeps the system prompt.

The OpenAI-compatible endpoint is `http://127.0.0.1:11434/v1`, and the model name is `humanlike-2.0-nvfp4`. Send `"reasoning_effort": "none"` for the fastest replies, because thinking is on by default.

## Repo layout

- `chat/` holds the chat page and a small Python server that forwards requests to Ollama.
- `ollama/Modelfile` defines the NVFP4 build.
- `models/` is ignored by git. Put the weights there.

## Notes

Run only one large model server at a time on the Mac. Ollama unloads the model after 5 idle minutes, and the next message then takes about 5 s longer.

The Modelfile uses Ollama's `qwen3.8` renderer, not the `chat_template.jinja` file from the upstream repo. The texting style carried over in a short test, but the two templates have not been compared.

## Credits

[LessThanThreeAI](https://huggingface.co/LessThanThreeAI) made the model and released it under Apache-2.0. The upstream card describes it as a finetune of `huihui-ai/Huihui-Qwen3.8-27B-abliterated`.
