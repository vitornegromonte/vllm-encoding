#!/usr/bin/env python3
"""
Encoder-only (família BERT) no vLLM.

Modelos encoder-only NÃO geram texto: produzem embeddings.
Por isso trocamos SamplingParams -> nada (LLM.embed cuida disso), llm.chat ->
llm.embed, e instanciamos o LLM com runner="pooling" (caminho de "pooling
models" do vLLM).

vllm==0.29.0: `LLM.encode` genérico agora exige `pooling_task` explícito
(embed/classify/token_embed/...); `LLM.embed(...)` é o atalho dedicado pra
embeddings e já cuida disso. Normalização também saiu de PoolingParams --
vira PoolerConfig.use_activation, detectado automaticamente do próprio
modelo pra arquiteturas sentence-transformers (bge-m3, BERTimbau, ...).
"""
import math
import os
import sys
from pathlib import Path

from vllm import LLM

# Qualquer BERT-family suportada pelo vLLM funciona, ex.:
#   google-bert/bert-base-uncased
#   neuralmind/bert-base-portuguese-cased (BERTimbau)
#   sentence-transformers/all-MiniLM-L6-v2
#   BAAI/bge-m3 (XLM-R, multilíngue — boa para PT-BR)
# Modelos com config do sentence-transformers (BGE, MiniLM, E5) já trazem o
# pooling correto (mean); BERT "cru" cai no default CLS + normalize.
MODEL_PATH = "BAAI/bge-m3"
DEVICE = "cuda:0"
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 512  # BERT-family: limite de 512 posições — nada de 3768


def resolve_prompt(arg: str) -> str:
    """Se `arg` for um arquivo existente, usa seu conteúdo; senão, o próprio texto."""
    path = Path(arg)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return arg


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python embed_vllm.py <prompt ou arquivo.txt> [segundo prompt]")
        sys.exit(1)

    prompts = [resolve_prompt(a) for a in sys.argv[1:]]

    # vLLM pina a GPU via CUDA_VISIBLE_DEVICES — setar antes de instanciar LLM(...).
    if DEVICE.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE.split(":", 1)[1]

    print(f"Carregando modelo encoder-only ({MODEL_PATH} via vLLM, device={DEVICE})...")
    llm = LLM(
        model=MODEL_PATH,
        runner="pooling",  # <- essencial; em versões antigas do vLLM use task="embed"
        dtype="auto",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
    )

    print(f"Encodando {len(prompts)} prompt(s)...\n")
    outputs = llm.embed(
        prompts,
        tokenization_kwargs={"truncation": True, "max_length": MAX_MODEL_LEN},
    )
    embeddings = [o.outputs.embedding for o in outputs]

    for prompt, emb in zip(prompts, embeddings):
        norm = math.sqrt(sum(x * x for x in emb))
        print(f"{'=' * 60}")
        print(f"Prompt ({len(prompt)} chars) -> embedding dim={len(emb)}, norm={norm:.4f}")
        print("primeiros 8 valores:", [round(x, 4) for x in emb[:8]])

    if len(embeddings) == 2:
        cos = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
        print(f"{'=' * 60}")
        print(f"Similaridade de cosseno entre os 2 prompts: {cos:.4f}")

    print(f"{'=' * 60}\nConcluído.")


if __name__ == "__main__":
    main()