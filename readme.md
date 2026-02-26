# GRPO on Graph Isomorphism: A Cold-Start Problem

**Homework 2** — Research in LLMs, HSE × Central University

## What this is

I applied GRPO to teach Qwen2.5-1.5B-Instruct to solve graph isomorphism. The base model has 0% accuracy on the task (it always outputs "NOT ISOMORPHIC" in 2 tokens), which means GRPO gets zero reward variance and zero gradient. I call this the cold-start problem.

Three strategies to break out of it (reward shaping, prompt hints, SFT warmup) each failed for different reasons. An edge counting positive control confirmed the training code works but diverged due to β=0.0. An eval bug silently skipped adapter loading during inference, invalidating all post-training accuracy measurements.

**[Read the report](report/LLMR_report_HW1.pdf)**

## Repo structure

```
├── report/
│   ├── report.tex
│   ├── report.pdf
│   └── figures/          # 4 training dynamics plots
├── src/
│   ├── graph_isomorphism_env.py
│   ├── edge_counting_env.py
│   ├── rewards.py
│   ├── edge_counting_reward.py
│   ├── train.py           # all run configs inside
│   ├── eval.py
│   ├── eval_ec.py
│   ├── sft_warmup.py
│   ├── callbacks.py       # collapse monitor
│   └── data_utils.py
├── test_sets/             # 54 frozen evaluation sets (~10k instances)
├── tests/                 # unit tests for verifier
├── outputs/               # saved adapters + training logs
│   ├── run1_antihack/
│   ├── run_h1_hints/
│   ├── run_s1_sft/
│   ├── run_s1_sft_grpo/
│   └── run_ec/
└── results/               # evaluation CSVs (all measure base model due to eval bug)
```

## How to run

**Environment** (tested on Ubuntu 22.04, RTX 4090, CUDA 12.1):
```bash
conda create -n hw2 python=3.10 -y && conda activate hw2
pip install torch==2.1.0 unsloth trl vllm networkx safetensors
```

**Generate test sets:**
```bash
python src/graph_isomorphism_env.py  # writes to test_sets/
```

**Train** (pick a config in `src/train.py`):
```bash
python src/train.py  # edit ACTIVE_CONFIG at the bottom
```

**Evaluate:**
```bash
python src/eval.py --adapter_path outputs/run1_antihack/checkpoint-200 --test_dir test_sets/
```

Note: the eval script in this repo has the adapter loading bug fixed.

## HuggingFace artifacts

| Artifact | Link |
|----------|------|
| SFT adapter | [Melodiz/qwen2.5-1.5b-gi-sft](https://huggingface.co/Melodiz/qwen2.5-1.5b-gi-sft) |
| GRPO adapter | [Melodiz/qwen2.5-1.5b-gi-grpo-antihack](https://huggingface.co/Melodiz/qwen2.5-1.5b-gi-grpo-antihack) |
| Test sets | [Melodiz/graph-isomorphism-test-sets](https://huggingface.co/datasets/Melodiz/graph-isomorphism-test-sets) |