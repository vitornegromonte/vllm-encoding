# -*- coding: utf-8 -*-
"""
Cruza os BOs reconstruídos (bos_reconstruidos_clean_b.jsonl) com as
entidades extraídas pelo NER real (gemma4, em
bos_reconstruidos_gemma4_e4b_preannotate_20260729_001/annotated_documents.jsonl)
e monta a lista de BOs no formato esperado por `MotorDeLinkage.adicionar_bo`
-- o mesmo formato usado pelos `bos_sinteticos` do notebook de validação
original (`id`, `texto`, `data_fato`, `municipio`, `natureza`, `entidades`).

Para os BOs sem anotação NER (~11% do total), cai no fallback: entidades
derivadas dos campos estruturados do próprio registro (pessoas,
documentos, objetos, naturezas) + regex sobre o histórico (cobre PLACA,
que não vem nem do NER nem dos campos estruturados de forma limpa).
"""
import json

from motor_linkage import extrair_entidades_por_regex

# Caminhos relativos à raiz do fork (refactor/)
CAMINHO_DADOS = "data/raw/bos_reconstruidos_clean_b.jsonl"
CAMINHO_NER = "data/raw/bos_reconstruidos_gemma4_e4b_preannotate_20260729_001/annotated_documents.jsonl"


def normaliza_municipio(m):
    return m.strip().upper() if m else None


def construir_geo_lookup(caminho=CAMINHO_DADOS):
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


def carregar_entidades_ner(caminho=CAMINHO_NER):
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


def entidades_fallback(d):
    """Deriva entidades dos campos estruturados do registro, usado só
    para BOs sem anotação NER."""
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


def bo_para_motor(d, entidades_ner):
    """Converte um registro de bos_reconstruidos_clean_b.jsonl pro
    formato esperado por MotorDeLinkage.adicionar_bo, cruzando com as
    entidades do NER real (fallback pros campos estruturados quando o
    BO não tem anotação)."""
    naturezas = d.get("naturezas") or []
    bo_id = d["bo_id"]

    texto_hist = d.get("historico_clean") or d.get("historico_raw") or ""
    placas = [e for e in extrair_entidades_por_regex(texto_hist) if e["tipo"] == "PLACA"]

    if bo_id in entidades_ner:
        entidades = entidades_ner[bo_id] + placas
    else:
        entidades = entidades_fallback(d)

    return {
        "id": bo_id,
        "texto": texto_hist,
        "data_fato": d.get("data_fato"),
        "municipio": normaliza_municipio(d.get("municipio")),
        "natureza": "/".join(naturezas),
        "entidades": entidades,
    }


def carregar_bos_reais(limite=None, apenas_ids=None, caminho=CAMINHO_DADOS, entidades_ner=None):
    """Lê bos_reconstruidos_clean_b.jsonl e devolve a lista de BOs já
    cruzada com o NER, no formato `bos_sinteticos` do notebook original.

    - `limite`: pega os primeiros N registros do arquivo (amostra).
    - `apenas_ids`: se passado, filtra só esses bo_ids (ignora `limite`).
    """
    if entidades_ner is None:
        entidades_ner = carregar_entidades_ner()

    alvo = set(apenas_ids) if apenas_ids else None
    bos = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            d = json.loads(linha)
            if alvo is not None:
                if d["bo_id"] in alvo:
                    bos.append(bo_para_motor(d, entidades_ner))
                    if len(bos) == len(alvo):
                        break
                continue
            bos.append(bo_para_motor(d, entidades_ner))
            if limite and len(bos) >= limite:
                break
    return bos
