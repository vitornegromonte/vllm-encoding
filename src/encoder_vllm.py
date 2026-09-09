# -*- coding: utf-8 -*-
"""
Adapter que expõe a MESMA interface do SentenceTransformer usada por
MotorDeLinkage (`.encode(lista_de_textos, convert_to_numpy=True) ->
np.ndarray`), mas gera os embeddings via vLLM (LLM.embed) em vez de
sentence_transformers puro -- não precisa mexer em motor_linkage.py.

Motivação: sentence_transformers roda o encoder síncrono, texto por
texto (ou em pequenos lotes manuais); vLLM faz batching/scheduling
contínuo na GPU, relevante quando o volume de encode cresce (ingestão
de milhares de BOs, cada um chamando .encode() várias vezes -- texto
geral + descrições).

Usa por padrão o checkpoint LOCAL convertido (ver
converter_checkpoint_bert.py) em vez do nome do repo HuggingFace
("neuralmind/bert-base-portuguese-cased") diretamente -- vLLM==0.29.0
não consegue tratar esse checkpoint como modelo de embedding sem
conversão prévia: a arquitetura declarada no config.json do repo é
BertForMaskedLM (mapeada por vLLM pra um modelo de token-classification,
não embedding), e mesmo forçando hf_overrides={"architectures":
["BertModel"]} os nomes dos pesos não batem (checkpoint salvo com
prefixo "bert." que a classe BertEmbeddingModel do vLLM não espera).
Rode `python converter_checkpoint_bert.py` uma vez antes de usar este
módulo pela primeira vez.

⚠️ Mantém a dimensão (768) e a família de modelo (BERTimbau) do índice
de produção original, mas o pooling exato (CLS via vLLM) não é
garantido bit-idêntico ao mean-pooling do sentence_transformers --
validado como semanticamente equivalente (mesma direção/magnitude de
similaridade em teste com textos de BO reais), não bit-exato. Qualquer
outro modelo BERT-family funciona (ver embed_vllm.py), mas trocar o
modelo muda o espaço vetorial: um índice salvo com um encoder não é
comparável com embeddings gerados por outro.
"""
import os

import numpy as np
from vllm import LLM

_DIR_DESTE_ARQUIVO = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_LOCAL_PADRAO = os.path.join(_DIR_DESTE_ARQUIVO, "..", "data", "external", "bertimbau-embed")


class EncoderVLLM:
    def __init__(self, model_path=CHECKPOINT_LOCAL_PADRAO,
                 max_model_len=512, gpu_memory_utilization=0.9, device="cuda:0"):
        if model_path == CHECKPOINT_LOCAL_PADRAO and not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Checkpoint local '{model_path}' não existe. Rode "
                "`python converter_checkpoint_bert.py` primeiro (converte "
                "neuralmind/bert-base-portuguese-cased pro formato que o "
                "vLLM reconhece como modelo de embedding -- ver docstring "
                "deste módulo)."
            )

        if device.startswith("cuda:"):
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", device.split(":", 1)[1])

        self.max_model_len = max_model_len
        self._llm = LLM(
            model=model_path,
            runner="pooling",
            # convert="auto" (default) só resolve pra embedding sozinho
            # quando o modelo já tem config nativa de sentence-transformers
            # (ex: bge-m3, que tem modules.json com pooling declarado) OU
            # quando architectures no config.json já é "BertModel" (caso
            # do checkpoint local convertido). Setar explícito de qualquer
            # forma, sem custo se já for auto-detectável.
            convert="embed",
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    def encode(self, textos, convert_to_numpy=True, **_ignorados):
        """Mesma assinatura de SentenceTransformer.encode -- **_ignorados
        absorve kwargs que motor_linkage.py não usa hoje (ex:
        batch_size), pra não quebrar se algum dia passarem a passar."""
        if not textos:
            return np.zeros((0, 0), dtype=np.float32) if convert_to_numpy else []

        outputs = self._llm.embed(
            list(textos),
            tokenization_kwargs={"truncation": True, "max_length": self.max_model_len},
        )
        embeddings = [o.outputs.embedding for o in outputs]
        return np.array(embeddings, dtype=np.float32) if convert_to_numpy else embeddings
