# Qwen3.8 Humanchat MLX

**TL;DR:** This repo serves [Qwen3.8-27B-Humanlike-Chat 2.0](https://huggingface.co/LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-2.0) on an Apple Silicon Mac through Ollama's MLX runner. It adds a local chat page with a system prompt and a benchmark script. On an M4 Max, an NVFP4 build with draft tokens decoded at about 2x the speed of the published Q4_K_M GGUF in llama.cpp.

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

## Results

The host was an Apple M4 Max with 128 GB of memory. A VM and desktop apps ran during the tests. Each llama.cpp row used a 512-token decode, and latency values are the median of 3 runs. The JSON files are in `results/`.

| Metric | GGUF Q8_0 | GGUF Q4_K_M | Q4_K_M, `-ub 128` | Ollama NVFP4 |
|---|---:|---:|---:|---:|
| Decode | 14.3 tok/s | 19.0 tok/s | 19.4 tok/s | **40.4 tok/s** |
| First token, 92 tokens | 0.58 s | 0.61 s | 0.61 s | 0.47 s |
| First token, 7.1k tokens, cold | 33.3 s | 35.0 s | 35.0 s | 28.4 s |
| First token, 7.1k tokens, warm | 3.1 s | 3.1 s | 1.14 s | **0.32 s** |
| First token, 17.6k tokens, cold | 87.8 s | 95.8 s | 95.2 s | 75.4 s |
| Total throughput, 4 streams | 23.6–25.1 tok/s | 21.7–21.9 tok/s | not measured | not measured |
| Memory | 48 GB | 36 GB | not measured | 32 GB |

The Ollama benchmark was stopped early. Its decode value comes from one 200-token test at temperature 0, so treat it as a single sample. Its draft acceptance was 0.56–0.67. One of its 3 warm runs took 29.9 s, so the prompt cache can miss. Tool calls and concurrency were not tested on this build.

Long prompts remain the main latency cost. Cold prompt speed stayed between 185 and 250 tok/s on every setup.

To repeat the tests, start the model, then run the script below. It uses the Ollama endpoint by default. Use `--base` and `--model` to test another OpenAI-compatible endpoint, and `--stages` to run only some tests.

```bash
python3 bench/bench.py --out results/my-run.json
```

## Repo layout

- `chat/` holds the chat page and a small Python server that forwards requests to Ollama.
- `bench/bench.py` measures first-token latency, decode speed, and concurrency on any OpenAI-compatible endpoint.
- `ollama/Modelfile` defines the NVFP4 build.
- `results/` holds the raw benchmark JSON.
- `models/` is ignored by git. Put the weights there.

## Notes

Run only one large model server at a time on the Mac. Ollama unloads the model after 5 idle minutes, and the next message then takes about 5 s longer.

The Modelfile uses Ollama's `qwen3.8` renderer, not the `chat_template.jinja` file from the upstream repo. The texting style carried over in a short test, but the two templates have not been compared.

## Credits

[LessThanThreeAI](https://huggingface.co/LessThanThreeAI) made the model and released it under Apache-2.0. The upstream card describes it as a finetune of `huihui-ai/Huihui-Qwen3.8-27B-abliterated`.
