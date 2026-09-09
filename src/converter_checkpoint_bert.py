# -*- coding: utf-8 -*-
"""
Converte um checkpoint BERT "cru" do HuggingFace (arquitetura
BertForMaskedLM, pesos com prefixo `bert.` + cabeça `cls.*` de MLM) pra
um checkpoint local que o vLLM reconhece como modelo de embedding
(`runner="pooling"`, `convert="embed"`).

Por quê isso é necessário: vLLM registra a arquitetura "BertModel" ->
BertEmbeddingModel (que sabe fazer embed), mas essa classe espera pesos
SEM o prefixo `bert.` (formato dos checkpoints google-bert/* "puros").
Checkpoints como neuralmind/bert-base-portuguese-cased foram salvos
como BertForMaskedLM completo (prefixo `bert.` nos pesos do encoder,
mais a cabeça `cls.*` de masked-LM) -- forçar `hf_overrides=
{"architectures": ["BertModel"]}` sem converter os pesos primeiro dá
"ValueError: There is no module or parameter named 'bert' in BertModel"
(vLLM==0.29.0).

A conversão aqui é offline e local, uma vez só: baixa o .bin original,
remove o prefixo `bert.` de cada tensor, descarta a cabeça `cls.*`
(não usada pra embedding), salva em safetensors, e escreve um
config.json com architectures=["BertModel"] pro vLLM já reconhecer
direto ao apontar pra esse diretório.

Validado (ver notas do fork): similaridade de cosseno entre pares de
texto batendo na mesma direção/magnitude que sentence_transformers
original rodando o mesmo checkpoint (0.96 vs 0.92 e 0.63 vs 0.63 num
teste com 3 textos de BO) -- não é bit-exact (pooling CLS via vLLM não
é garantido idêntico ao mean-pooling do sentence_transformers), mas é
semanticamente equivalente.

Uso:
    python converter_checkpoint_bert.py [repo_hf] [dir_saida]
    # default: neuralmind/bert-base-portuguese-cased -> data/bertimbau-embed
"""
import json
import os
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REPO_PADRAO = "neuralmind/bert-base-portuguese-cased"
DIR_SAIDA_PADRAO = "data/external/bertimbau-embed"

ARQUIVOS_AUXILIARES = [
    "config.json", "vocab.txt", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
]


def converter(repo=REPO_PADRAO, dir_saida=DIR_SAIDA_PADRAO):
    os.makedirs(dir_saida, exist_ok=True)

    print(f"Baixando pesos de {repo}...")
    path = hf_hub_download(repo, "pytorch_model.bin")
    sd = torch.load(path, map_location="cpu", weights_only=True)

    novo_sd = {}
    for k, v in sd.items():
        if k.startswith("cls."):
            continue  # cabeça MLM -- não usada pra embedding
        if not k.startswith("bert."):
            raise ValueError(
                f"peso inesperado sem prefixo 'bert.': {k!r} -- "
                "checkpoint pode já estar num formato diferente do esperado"
            )
        novo_sd[k[len("bert."):]] = v.contiguous()

    print(f"  {len(novo_sd)} tensores mantidos (de {len(sd)} originais, "
          f"{len(sd) - len(novo_sd)} descartados da cabeça MLM)")
    save_file(novo_sd, os.path.join(dir_saida, "model.safetensors"))

    for fname in ARQUIVOS_AUXILIARES:
        try:
            p = hf_hub_download(repo, fname)
            shutil.copy(p, os.path.join(dir_saida, fname))
        except Exception as e:
            print(f"  aviso: não consegui copiar {fname}: {e}")

    caminho_config = os.path.join(dir_saida, "config.json")
    with open(caminho_config, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["architectures"] = ["BertModel"]
    with open(caminho_config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"Convertido em '{dir_saida}' -- {os.listdir(dir_saida)}")
    return dir_saida


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else REPO_PADRAO
    dir_saida = sys.argv[2] if len(sys.argv) > 2 else DIR_SAIDA_PADRAO
    converter(repo, dir_saida)
