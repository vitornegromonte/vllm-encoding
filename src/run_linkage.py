# -*- coding: utf-8 -*-
"""
Fork de run_linkage.py (ver ../../src/run_linkage.py) usando vLLM como
encoder em vez de sentence_transformers puro -- via EncoderVLLM
(encoder_vllm.py), que expõe a mesma interface .encode(...) que
MotorDeLinkage espera, então nada em motor_linkage.py muda.

⚠️ Isso muda o espaço vetorial dos embeddings: um índice gerado aqui
NÃO é compatível/comparável com indice_producao_* do projeto original
(mesmo usando o mesmo modelo nominal -- backend de inferência diferente
pode não ser bit-exact). Rodar do zero pra gerar um índice próprio
neste fork.

Executa o motor de linkage sobre bos_reconstruidos_clean_b.jsonl.

Fonte primária de entidades: NER real (gemma4) em
bos_reconstruidos_gemma4_e4b_preannotate_20260729_001/annotated_documents.jsonl
-- cobre 10.518 dos 11.838 BOs, nos 34 tipos do SCHEMA_ENTIDADES.
Para os BOs sem anotação NER (~1.320), cai no fallback: entidades
derivadas dos campos estruturados do próprio registro (pessoas,
documentos, objetos, naturezas) + regex sobre o histórico (cobre PLACA,
que não vem nem do NER nem dos campos estruturados de forma limpa).
"""
import json
import sys
import time

import joblib

from motor_linkage import MotorDeLinkage, extrair_entidades_por_regex
from encoder_vllm import EncoderVLLM

# Caminhos relativos à raiz do fork (refactor/) -- rodar como:
#   cd refactor && python src/run_linkage.py
CAMINHO_DADOS = "data/raw/bos_reconstruidos_clean_b.jsonl"
CAMINHO_NER = "data/raw/bos_reconstruidos_gemma4_e4b_preannotate_20260729_001/annotated_documents.jsonl"
CAMINHO_CLASSIFICADOR = "models/modelo_v1_classificador.joblib"
PREFIXO_INDICE = "data/processed/indice_producao"


def carregar_entidades_ner(caminho):
    """{doc_id: [{"texto": ..., "tipo": ...}, ...]}"""
    por_doc = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            d = json.loads(linha)
            por_doc[d["doc_id"]] = [
                {"texto": e["text"], "tipo": e["label"]} for e in d.get("entities", [])
            ]
    return por_doc


def normaliza_municipio(m):
    return m.strip().upper() if m else None


def construir_geo_lookup(caminho):
    """Usa lat/lon média por município a partir dos próprios registros
    que já trazem coordenada -- não há base externa de geocodificação
    disponível neste pacote."""
    soma = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            d = json.loads(linha)
            mun = normaliza_municipio(d.get("municipio"))
            try:
                lat = float(d.get("latitude"))
                lon = float(d.get("longitude"))
            except (TypeError, ValueError):
                continue
            if mun and lat != 0.0 and lon != 0.0:
                s = soma.setdefault(mun, [0.0, 0.0, 0])
                s[0] += lat
                s[1] += lon
                s[2] += 1
    return {mun: (s[0] / s[2], s[1] / s[2]) for mun, s in soma.items()}


def entidades_do_bo(d):
    entidades = []

    for nat in d.get("naturezas") or []:
        if nat and nat.strip():
            entidades.append({"texto": nat, "tipo": "OCORRENCIA"})

    for p in d.get("pessoas") or []:
        nome = p.get("nome")
        if nome and nome.strip() and p.get("tipo_pessoa") == "Física":
            entidades.append({"texto": nome, "tipo": "PESSOA_NOME"})
        elif nome and nome.strip():
            entidades.append({"texto": nome, "tipo": "PESSOA_MENCAO"})

        for doc in p.get("documentos") or []:
            numero = doc.get("numero")
            if numero:
                entidades.append({"texto": str(numero), "tipo": "DOCUMENTO"})

        if p.get("celular"):
            entidades.append({"texto": str(p["celular"]), "tipo": "TELEFONE"})
        if p.get("imei"):
            entidades.append({"texto": str(p["imei"]), "tipo": "DOCUMENTO"})

        for obj in p.get("objetos") or []:
            texto_obj = " ".join(
                str(obj.get(k, "")) for k in ("categoria", "marca", "modelo", "descricao")
            ).strip()
            if not texto_obj:
                continue
            tipo_obj = (obj.get("tipo_objeto") or "").upper()
            if tipo_obj == "VEICULO":
                entidades.append({"texto": texto_obj, "tipo": "OBJETO_VEICULO"})
            elif tipo_obj in ("ARMA DE FOGO", "ARMA BRANCA"):
                entidades.append({"texto": texto_obj, "tipo": "OBJETO_ARMA"})
            elif tipo_obj == "ENTORPECENTE":
                entidades.append({"texto": texto_obj, "tipo": "ENTORPECENTE"})
            else:
                entidades.append({"texto": texto_obj, "tipo": "OBJETO_GERAL"})

    for obj in d.get("objetos_sem_pessoa") or []:
        texto_obj = " ".join(
            str(obj.get(k, "")) for k in ("categoria", "marca", "modelo", "descricao")
        ).strip()
        if not texto_obj:
            continue
        tipo_obj = (obj.get("tipo_objeto") or "").upper()
        if tipo_obj == "VEICULO":
            entidades.append({"texto": texto_obj, "tipo": "OBJETO_VEICULO"})
        elif tipo_obj in ("ARMA DE FOGO", "ARMA BRANCA"):
            entidades.append({"texto": texto_obj, "tipo": "OBJETO_ARMA"})
        else:
            entidades.append({"texto": texto_obj, "tipo": "OBJETO_GERAL"})

    if d.get("bairro"):
        entidades.append({"texto": d["bairro"], "tipo": "LOCAL_AREA"})
    if d.get("endereco"):
        entidades.append({"texto": d["endereco"], "tipo": "LOCAL_ESTABELECIMENTO"})

    texto_hist = d.get("historico_clean") or d.get("historico_raw") or ""
    entidades.extend(extrair_entidades_por_regex(texto_hist))

    return entidades


def bo_para_motor(d):
    naturezas = d.get("naturezas") or []
    return {
        "id": d["bo_id"],
        "texto": d.get("historico_clean") or d.get("historico_raw") or "",
        "data_fato": d.get("data_fato"),
        "municipio": normaliza_municipio(d.get("municipio")),
        "natureza": "/".join(naturezas),
    }


def main():
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else None

    print("Construindo geo_lookup a partir dos próprios registros...")
    geo_lookup = construir_geo_lookup(CAMINHO_DADOS)
    print(f"  {len(geo_lookup)} municípios com coordenada.")

    print(f"Carregando entidades NER (gemma4) de {CAMINHO_NER}...")
    entidades_ner = carregar_entidades_ner(CAMINHO_NER)
    print(f"  {len(entidades_ner)} BOs com anotação NER.")

    print("Carregando encoder via vLLM (BERTimbau, checkpoint local convertido)...")
    t0 = time.time()
    encoder = EncoderVLLM()
    print(f"  ok ({time.time() - t0:.1f}s)")

    clf = joblib.load(CAMINHO_CLASSIFICADOR)

    motor = MotorDeLinkage(
        encoder, geo_lookup, clf=clf,
        tau_geo_km=15.0,
        tau_dias=30.0,
        limiar_descricao=0.6,
    )

    n_ner, n_fallback = 0, 0
    print(f"Ingerindo BOs de {CAMINHO_DADOS}" + (f" (limite={limite})" if limite else "") + "...")
    t0 = time.time()
    n = 0
    with open(CAMINHO_DADOS, encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            d = json.loads(linha)
            bo = bo_para_motor(d)

            texto_hist = d.get("historico_clean") or d.get("historico_raw") or ""
            placas = extrair_entidades_por_regex(texto_hist)
            placas = [e for e in placas if e["tipo"] == "PLACA"]

            if d["bo_id"] in entidades_ner:
                entidades = entidades_ner[d["bo_id"]] + placas
                n_ner += 1
            else:
                entidades = entidades_do_bo(d)
                n_fallback += 1

            motor.adicionar_bo(bo, entidades=entidades)
            n += 1
            if n % 500 == 0:
                print(f"  {n} BOs ingeridos... ({time.time() - t0:.1f}s)")
            if limite and n >= limite:
                break
    print(f"Ingestão concluída: {n} BOs em {time.time() - t0:.1f}s "
          f"({n_ner} via NER, {n_fallback} via fallback estruturado)")

    print(f"Salvando índice em '{PREFIXO_INDICE}_*'...")
    motor.salvar(PREFIXO_INDICE)
    print("Salvo.")

    return motor


if __name__ == "__main__":
    main()
