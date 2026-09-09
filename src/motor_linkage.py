# -*- coding: utf-8 -*-
"""
Motor de linkage pra produção -- o que roda por trás da interface do
policial. Três peças, porque as 3 perguntas do seu caso de uso NÃO são
a mesma pergunta:

  1) "Dado esse BO, quais estão conectados a ele?"
     -> retrieval: score do BO consulta contra todo o índice já existente.

  2) "Dado esse tipo de crime, quais grupos de BOs se conectam entre si?"
     -> filtra pela entidade OCORRENCIA (nome do crime, extraída pelo
        NER -- mais limpa que o campo 'natureza' livre; cai pro campo
        'natureza' como fallback se a entidade não estiver disponível)
        e dentro do subconjunto monta um grafo de conexão -> componentes.

  3) "Essa pessoa tem mais de um BO?"
     -> resolução de entidade: nome normalizado + fuzzy match contra as
        entidades PESSOA_NOME (nunca PESSOA_MENCAO -- "vítima"/"condutor"
        não identificam ninguém especificamente).

MENÇÃO INDIRETA (o motivo de PESSOA_DESCRICAO/OBJETO_* existirem): dois
BOs podem estar ligados sem nome, placa ou documento em comum -- só por
uma descrição distintiva repetida ("arma rosa", "moto com adesivo de
pantera", "homem alto com tatuagem"). Isso não dá pra comparar por
igualdade exata (a redação varia: "arma rosa" vs "pistola cor-de-rosa"),
então entra como um sinal FUZZY/SEMÂNTICO à parte (sim_descricao),
usando o mesmo encoder de texto pra comparar as descrições entre si.

PERSISTÊNCIA: encoder fixo, classificador salvo via joblib, índice de
embeddings/entidades cresce incrementalmente.

CACHE DE SCORES: como os dados de um BO já inserido não mudam (só
existe adicionar_bo, nunca editar_bo), o score de um par (A, B) uma vez
calculado nunca fica desatualizado. self.cache_scores guarda esses
scores por frozenset({id_a, id_b}) -- evita recalcular O(n^2) pares que
já foram vistos em chamadas anteriores (ex: clusters_por_natureza
chamado de novo após a chegada de um BO novo). O cache é persistido
junto no salvar()/carregar(). Se no futuro existir edição de BO, será
necessário invalidar as entradas do cache que envolvem aquele bo_id.
"""
import json
import re
import unicodedata
import difflib
from datetime import datetime
from math import radians, sin, cos, asin, sqrt

import numpy as np
import joblib
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# SCHEMA DE ENTIDADES (Guia 0 — 34 tipos; ordem importa pro perfil)
# ------------------------------------------------------------
SCHEMA_ENTIDADES = [
    "PESSOA_NOME", "PESSOA_DESCRICAO", "PESSOA_MENCAO", "ORGAO", "CARGO",
    "DOCUMENTO", "OBJETO_DOCUMENTO", "PLACA", "BO_ID", "DATA", "HORA",
    "VALOR_MONETARIO", "VALOR_MEDIDA", "VALOR_TEMPORAL",
    "EMAIL", "TELEFONE", "INSTAGRAM", "URL",
    "LOCAL_AREA", "LOCAL_ESTABELECIMENTO", "LOCAL_COMODO",
    "OBJETO_GERAL", "OBJETO_VEICULO", "OBJETO_ARMA",
    "ENTORPECENTE", "OBJETO_ENTORPECENTE", "OCORRENCIA", "ARTIGO_LEI",
    "CONTA_BANCARIA", "AGENCIA_BANCARIA", "DOCUMENTO_BOLETO",
    "CHAVE_PIX", "TRANSACAO_FINANCEIRA",
    "BO_COMPLEMENTADO",
]
N_ENT_TIPOS = len(SCHEMA_ENTIDADES)
ENT2IDX = {e: i for i, e in enumerate(SCHEMA_ENTIDADES)}

# Identifica QUEM/O QUÊ especificamente -- exato, forte.
ENTIDADES_ID = {
    "PESSOA_NOME", "DOCUMENTO", "PLACA", "BO_ID",
    "EMAIL", "TELEFONE", "INSTAGRAM", "URL",
    "CONTA_BANCARIA", "CHAVE_PIX", "OBJETO_VEICULO",
}
# Local/tempo -- menos discriminativo sozinho.
ENTIDADES_CONTEXTO = {
    "LOCAL_AREA", "LOCAL_ESTABELECIMENTO", "LOCAL_COMODO",
    "DATA", "HORA", "VALOR_TEMPORAL", "AGENCIA_BANCARIA",
}
# Descreve O QUE aconteceu -- não identifica QUEM.
ENTIDADES_EVENTO = {
    "OCORRENCIA", "ENTORPECENTE", "OBJETO_ENTORPECENTE", "ARTIGO_LEI",
    "OBJETO_GERAL", "CARGO", "ORGAO", "OBJETO_ARMA",
    "VALOR_MONETARIO", "VALOR_MEDIDA", "TRANSACAO_FINANCEIRA",
    "DOCUMENTO_BOLETO", "OBJETO_DOCUMENTO", "BO_COMPLEMENTADO",
}

TIPOS_LOCAL_EXATO = {"LOCAL_AREA", "LOCAL_ESTABELECIMENTO", "LOCAL_COMODO"}
IDENTIFICADORES_LINKAGE = set(ENTIDADES_ID)

# Tipos usados no sinal de DESCRIÇÃO INDIRETA (fuzzy/semântico, não
# exato). PESSOA_DESCRICAO é o caso central ("homem alto com tatuagem"),
# mas OBJETO_GERAL/OBJETO_ARMA/OBJETO_VEICULO também carregam descrição
# distintiva quando o texto tem modificador incomum ("arma rosa", "moto
# com adesivo de pantera") -- mesmo já usando OBJETO_VEICULO como
# identificador exato acima, aqui ele entra TAMBÉM como fuzzy, pra
# pegar o caso em que a descrição bate mas a redação varia.
TIPOS_DESCRITIVOS_PADRAO = {"PESSOA_DESCRICAO", "OBJETO_GERAL", "OBJETO_ARMA", "OBJETO_VEICULO"}

# ------------------------------------------------------------
# Utilitários gerais
# ------------------------------------------------------------
def haversine_km(p1, p2):
    lat1, lon1, lat2, lon2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))

def normalizar_nome(nome):
    nome = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return " ".join(nome.upper().split())

def normalizar_natureza(natureza):
    nat = unicodedata.normalize("NFKD", natureza).encode("ascii", "ignore").decode()
    return nat.lower().strip()

def naturezas_compativeis(natureza_a, natureza_b):
    a, b = normalizar_natureza(natureza_a), normalizar_natureza(natureza_b)
    return a in b or b in a

_TIPOS_PESSOA = {"PESSOA_NOME"}
_TIPOS_IDENTIFICADOR_DURO = {
    "DOCUMENTO", "PLACA", "BO_ID", "TELEFONE",
    "CONTA_BANCARIA", "AGENCIA_BANCARIA", "CHAVE_PIX",
}

def normalizar_valor_entidade(texto, tipo):
    t = tipo.upper()
    if t in _TIPOS_PESSOA:
        return normalizar_nome(texto)
    if t == "EMAIL":
        return texto.strip().lower()
    if t == "URL":
        return texto.strip().lower().rstrip("/")
    if t == "INSTAGRAM":
        return texto.strip().lower().lstrip("@")
    if t in _TIPOS_IDENTIFICADOR_DURO:
        return re.sub(r"[^A-Z0-9]", "", texto.upper())
    txt = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^\w\s]", " ", txt.lower())
    return " ".join(txt.split())

# ------------------------------------------------------------
# Fallback SEM NER (regex bruto, bem mais limitado)
# ------------------------------------------------------------
_RE_PLACA = re.compile(r"\b[A-Z]{3}-?\d[A-Z0-9]\d{2}\b", re.IGNORECASE)
_RE_CPF = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_RE_IMEI = re.compile(r"\b\d{15}\b")
_RE_TELEFONE = re.compile(r"\(\d{2}\)\s?\d{4,5}-?\d{4}")

def extrair_entidades_por_regex(texto):
    entidades = []
    for regex, tipo in [(_RE_PLACA, "PLACA"), (_RE_CPF, "DOCUMENTO"),
                         (_RE_IMEI, "DOCUMENTO"), (_RE_TELEFONE, "TELEFONE")]:
        for m in regex.findall(texto):
            entidades.append({"texto": m, "tipo": tipo})
    return entidades

def ajustar_geo_data(sim_texto, sim_geo_bruta, sim_data_bruta, piso_relevancia):
    return sim_geo_bruta * piso_relevancia, sim_data_bruta * piso_relevancia

def perfil_tipos(entidades_por_tipo):
    v = np.zeros(N_ENT_TIPOS)
    for tipo in entidades_por_tipo:
        if tipo in ENT2IDX:
            v[ENT2IDX[tipo]] = 1.0
    norma = np.linalg.norm(v)
    return v / norma if norma > 0 else v

# ------------------------------------------------------------
# MOTOR DE LINKAGE
# ------------------------------------------------------------
class MotorDeLinkage:
    def __init__(self, encoder, geo_lookup, clf=None, tau_geo_km=15.0, tau_dias=30.0,
                 tipos_fortes=None, tipos_local=None, tipos_descritivos=None,
                 limiar_descricao=0.6):
        self.encoder = encoder
        self.geo_lookup = geo_lookup
        self.clf = clf
        self.tau_geo_km = tau_geo_km
        self.tau_dias = tau_dias
        self.tipos_fortes = set(tipos_fortes) if tipos_fortes else set(IDENTIFICADORES_LINKAGE)
        self.tipos_local = set(tipos_local) if tipos_local else set(TIPOS_LOCAL_EXATO)
        self.tipos_descritivos = set(tipos_descritivos) if tipos_descritivos else set(TIPOS_DESCRITIVOS_PADRAO)
        self.limiar_descricao = limiar_descricao

        self.ids = []
        self.emb_texto = None
        self.coords = []
        self.datas = []
        self.metadados = {}
        self.entidades_por_bo = {}          # {bo_id: {tipo: set(valores_normalizados)}}
        self.perfis = {}                    # {bo_id: vetor multi-hot, N_ENT_TIPOS}
        self.descricoes_por_bo = {}         # {bo_id: [(tipo, texto_original), ...]}
        self.emb_descricoes_por_bo = {}      # {bo_id: np.array (k, dim)}
        self.cache_scores = {}              # {frozenset({id_a, id_b}): score}

    # -------------------- ingestão --------------------
    def adicionar_bo(self, bo, entidades=None, calcular_scores_imediatamente=False):
        """
        calcular_scores_imediatamente=True (opção "eager"): assim que o
        BO entra, já calcula e cacheia o score dele contra TODOS os BOs
        já existentes no índice. Deixa adicionar_bo mais lento (O(n) em
        vez de O(1) -- compara contra todo mundo que já estava lá), mas
        a próxima consulta envolvendo esse BO (score_par, bos_conectados,
        clusters_por_natureza...) sai instantânea, já vem do cache.

        Use True quando BOs chegam um de cada vez e a interface do
        policial precisa responder rápido logo após o cadastro. Use o
        default False (lazy) para ingestão em lote -- senão inserir N
        BOs de uma vez volta a custar O(n^2) escondido dentro do loop.
        """
        if bo["id"] in self.ids:
            raise ValueError(f"BO {bo['id']} já está no índice")

        vec = self.encoder.encode([bo["texto"]], convert_to_numpy=True)[0]

        municipio = bo.get("municipio")
        if municipio and municipio in self.geo_lookup:
            coord = self.geo_lookup[municipio]
        else:
            coord = None  # sem município/coordenada -- bloco geo fica neutro pra esse BO

        data_str = bo.get("data_fato") or bo.get("data")
        data = datetime.strptime(data_str, "%d/%m/%Y") if data_str else None

        self.ids.append(bo["id"])
        self.emb_texto = vec[None, :] if self.emb_texto is None else np.vstack([self.emb_texto, vec])
        self.coords.append(coord)
        self.datas.append(data)
        self.metadados[bo["id"]] = bo

        if entidades is None:
            entidades = extrair_entidades_por_regex(bo["texto"])

        por_tipo = {}
        descritivas = []
        for ent in entidades:
            tipo = ent["tipo"].upper()
            if tipo not in ENT2IDX:
                continue
            valor = normalizar_valor_entidade(ent["texto"], tipo)
            if valor:
                por_tipo.setdefault(tipo, set()).add(valor)
            if tipo in self.tipos_descritivos:
                descritivas.append((tipo, ent["texto"]))

        self.entidades_por_bo[bo["id"]] = por_tipo
        self.perfis[bo["id"]] = perfil_tipos(por_tipo)
        self.descricoes_por_bo[bo["id"]] = descritivas
        if descritivas:
            self.emb_descricoes_por_bo[bo["id"]] = self.encoder.encode(
                [t for _, t in descritivas], convert_to_numpy=True
            )
        else:
            dim = self.emb_texto.shape[1]
            self.emb_descricoes_por_bo[bo["id"]] = np.zeros((0, dim))

        if calcular_scores_imediatamente:
            for outro_id in self.ids:
                if outro_id != bo["id"]:
                    self.score_par(bo["id"], outro_id)  # calcula e já cacheia

    def carregar_lote(self, bos, extrator_entidades=None):
        for bo in bos:
            entidades = extrator_entidades(bo) if extrator_entidades else None
            self.adicionar_bo(bo, entidades=entidades)

    # -------------------- score de par --------------------
    def _overlap_por_tipo(self, i, j, tipos):
        ent_i = self.entidades_por_bo.get(self.ids[i], {})
        ent_j = self.entidades_por_bo.get(self.ids[j], {})
        overlaps = {}
        for tipo in tipos:
            vi, vj = ent_i.get(tipo), ent_j.get(tipo)
            if vi and vj:
                overlaps[tipo] = 1.0 if (vi & vj) else 0.0
        return overlaps

    def _sim_descricao(self, i, j):
        """Maior similaridade (cosseno) entre qualquer descrição de i e
        qualquer descrição de j -- pega o melhor par, não a média (uma
        descrição batendo já é o suficiente pra levantar suspeita)."""
        emb_i = self.emb_descricoes_por_bo.get(self.ids[i])
        emb_j = self.emb_descricoes_por_bo.get(self.ids[j])
        if emb_i is None or emb_j is None or len(emb_i) == 0 or len(emb_j) == 0:
            return 0.0, None
        sims = cosine_similarity(emb_i, emb_j)
        pos = np.unravel_index(np.argmax(sims), sims.shape)
        melhor = float(sims[pos])
        par_textos = (self.descricoes_por_bo[self.ids[i]][pos[0]][1],
                      self.descricoes_por_bo[self.ids[j]][pos[1]][1])
        return melhor, par_textos

    def _componentes(self, i, j):
        sim_texto = float(cosine_similarity(self.emb_texto[i:i + 1], self.emb_texto[j:j + 1])[0, 0])

        if self.coords[i] is not None and self.coords[j] is not None:
            d_geo = haversine_km(self.coords[i], self.coords[j])
            sim_geo_bruta = float(np.exp(-d_geo / self.tau_geo_km))
        else:
            sim_geo_bruta = 0.0  # sem município pra algum dos dois -- bloco neutro, não penaliza nem beneficia

        if self.datas[i] is not None and self.datas[j] is not None:
            d_dias = abs((self.datas[i] - self.datas[j]).days)
            sim_data_bruta = float(np.exp(-d_dias / self.tau_dias))
        else:
            sim_data_bruta = 0.0

        overlaps_id = self._overlap_por_tipo(i, j, self.tipos_fortes)
        sim_identificador = max(overlaps_id.values(), default=0.0)

        overlaps_local = self._overlap_por_tipo(i, j, self.tipos_local)
        sim_local_exato = max(overlaps_local.values(), default=0.0)

        sim_descricao, par_descricao = self._sim_descricao(i, j)

        pi, pj = self.perfis.get(self.ids[i]), self.perfis.get(self.ids[j])
        sim_perfil = float(np.dot(pi, pj)) if pi is not None and pj is not None else 0.0

        motivos = [t for t, v in {**overlaps_id, **overlaps_local}.items() if v == 1.0]
        if par_descricao and sim_descricao >= self.limiar_descricao:
            motivos.append(f"descrição parecida ('{par_descricao[0]}' ~ '{par_descricao[1]}')")

        descricao_conta = sim_descricao if sim_descricao >= self.limiar_descricao else 0.0
        piso_relevancia = max(sim_texto, sim_identificador, sim_local_exato, descricao_conta)
        sim_geo, sim_data = ajustar_geo_data(sim_texto, sim_geo_bruta, sim_data_bruta, piso_relevancia)

        return sim_texto, sim_geo, sim_data, sim_identificador, sim_local_exato, sim_descricao, sim_perfil, motivos

    def _feats(self, i, j):
        sim_texto, sim_geo, sim_data, sim_id, sim_local, sim_desc, sim_perfil, _ = self._componentes(i, j)
        return [sim_texto, sim_geo, sim_data, sim_id, sim_local, sim_desc, sim_perfil]

    def score_par(self, bo_id_a, bo_id_b):
        chave = frozenset({bo_id_a, bo_id_b})
        if chave in self.cache_scores:
            return self.cache_scores[chave]

        i, j = self.ids.index(bo_id_a), self.ids.index(bo_id_b)
        feats = self._feats(i, j)
        score = float(self.clf.predict_proba([feats])[0][1]) if self.clf is not None else float(np.mean(feats))

        self.cache_scores[chave] = score
        return score

    def invalidar_cache_do_bo(self, bo_id):
        """Remove do cache todos os pares que envolvem bo_id. Só é
        necessário se algum dia existir edição/reprocessamento de um BO
        já inserido -- hoje adicionar_bo bloqueia reinserção do mesmo
        id, então o cache nunca fica desatualizado sozinho."""
        self.cache_scores = {
            chave: v for chave, v in self.cache_scores.items() if bo_id not in chave
        }

    # -------------------- CASO DE USO 1 --------------------
    def bos_conectados(self, bo_id_consulta, k=10, threshold=0.5):
        i = self.ids.index(bo_id_consulta)
        resultado = []
        for j, outro_id in enumerate(self.ids):
            if outro_id == bo_id_consulta:
                continue
            sim_texto, sim_geo, sim_data, sim_id, sim_local, sim_desc, sim_perfil, motivos = self._componentes(i, j)
            score = self.score_par(bo_id_consulta, outro_id)  # usa/preenche o cache
            resultado.append({
                "bo_id": outro_id, "score": float(score),
                "sim_texto": round(sim_texto, 3), "sim_geo": round(sim_geo, 3),
                "sim_data": round(sim_data, 3), "sim_local_exato": round(sim_local, 3),
                "sim_descricao": round(sim_desc, 3),
                "motivo": ("mesmo(a) " + "; ".join(motivos)) if motivos else "similaridade semântica",
            })
        resultado.sort(key=lambda x: -x["score"])
        acima_threshold = [r for r in resultado if r["score"] >= threshold]
        return acima_threshold, resultado[:k]

    # -------------------- CASO DE USO 2 --------------------
    def _ocorrencias_do_bo(self, bo_id):
        """Valores normalizados de OCORRENCIA extraídos pelo NER; se o
        BO não tiver essa entidade (NER não achou, ou fallback regex),
        cai pro campo estruturado 'natureza', partido por '/'."""
        valores = self.entidades_por_bo.get(bo_id, {}).get("OCORRENCIA", set())
        if valores:
            return valores
        nat = self.metadados[bo_id].get("natureza", "")
        return {normalizar_natureza(p) for p in nat.split("/") if p.strip()}

    def clusters_por_natureza(self, natureza, threshold=0.5):
        """Filtra pela entidade OCORRENCIA (nome do crime, extraída pelo
        NER -- mais limpa que o campo livre) com fallback pro campo
        'natureza' quando a entidade não estiver disponível. Depois monta
        grafo de conexão + componentes dentro do subconjunto.

        Reaproveita o cache_scores: pares já calculados em chamadas
        anteriores (ou pré-calculados via calcular_scores_imediatamente
        no adicionar_bo) não são recomputados."""
        alvo = normalizar_natureza(natureza)
        subset_ids = [id_ for id_ in self.ids
                      if any(alvo in v or v in alvo for v in self._ocorrencias_do_bo(id_))]
        G = nx.Graph()
        G.add_nodes_from(subset_ids)
        for a in range(len(subset_ids)):
            for b in range(a + 1, len(subset_ids)):
                s = self.score_par(subset_ids[a], subset_ids[b])
                if s >= threshold:
                    G.add_edge(subset_ids[a], subset_ids[b], weight=s)
        grupos = [sorted(c) for c in nx.connected_components(G)]
        grupos.sort(key=len, reverse=True)
        return grupos

    # -------------------- CASO DE USO 3 --------------------
    def bos_da_pessoa(self, nome_consulta, limiar=0.85):
        alvo = normalizar_nome(nome_consulta)
        candidatos = []
        for bo_id, por_tipo in self.entidades_por_bo.items():
            nomes = por_tipo.get("PESSOA_NOME", set())
            if not nomes:
                continue
            melhor = max(difflib.SequenceMatcher(None, alvo, n).ratio() for n in nomes)
            if melhor >= limiar:
                candidatos.append({"bo_id": bo_id, "score": round(melhor, 3),
                                    "natureza": self.metadados[bo_id].get("natureza")})
        candidatos.sort(key=lambda x: -x["score"])
        return candidatos

    # -------------------- persistência --------------------
    def salvar(self, caminho_prefixo):
        joblib.dump(self.clf, f"{caminho_prefixo}_classificador.joblib")
        np.save(f"{caminho_prefixo}_embeddings.npy", self.emb_texto)
        with open(f"{caminho_prefixo}_indice.json", "w", encoding="utf-8") as f:
            json.dump({
                "ids": self.ids,
                "coords": self.coords,
                "datas": [d.strftime("%d/%m/%Y") if d is not None else None for d in self.datas],
                "metadados": self.metadados,
                "entidades_por_bo": {
                    bo_id: {tipo: list(vals) for tipo, vals in por_tipo.items()}
                    for bo_id, por_tipo in self.entidades_por_bo.items()
                },
                "perfis": {bo_id: v.tolist() for bo_id, v in self.perfis.items()},
                "descricoes_por_bo": self.descricoes_por_bo,
                "emb_descricoes_por_bo": {
                    bo_id: v.tolist() for bo_id, v in self.emb_descricoes_por_bo.items()
                },
                "tipos_fortes": list(self.tipos_fortes),
                "tipos_local": list(self.tipos_local),
                "tipos_descritivos": list(self.tipos_descritivos),
                "limiar_descricao": self.limiar_descricao,
                "cache_scores": {
                    "||".join(sorted(chave)): valor
                    for chave, valor in self.cache_scores.items()
                },
            }, f, ensure_ascii=False)

    @classmethod
    def carregar(cls, caminho_prefixo, encoder, geo_lookup, tau_geo_km=15.0, tau_dias=30.0):
        clf = joblib.load(f"{caminho_prefixo}_classificador.joblib")
        with open(f"{caminho_prefixo}_indice.json", encoding="utf-8") as f:
            d = json.load(f)
        motor = cls(encoder, geo_lookup, clf=clf, tau_geo_km=tau_geo_km, tau_dias=tau_dias,
                    tipos_fortes=d.get("tipos_fortes"), tipos_local=d.get("tipos_local"),
                    tipos_descritivos=d.get("tipos_descritivos"),
                    limiar_descricao=d.get("limiar_descricao", 0.6))
        motor.emb_texto = np.load(f"{caminho_prefixo}_embeddings.npy")
        motor.ids = d["ids"]
        motor.coords = [tuple(c) if c is not None else None for c in d["coords"]]
        motor.datas = [datetime.strptime(s, "%d/%m/%Y") if s is not None else None for s in d["datas"]]
        motor.metadados = d["metadados"]
        motor.entidades_por_bo = {
            bo_id: {tipo: set(vals) for tipo, vals in por_tipo.items()}
            for bo_id, por_tipo in d["entidades_por_bo"].items()
        }
        motor.perfis = {bo_id: np.array(v) for bo_id, v in d["perfis"].items()}
        motor.descricoes_por_bo = {k: [tuple(x) for x in v] for k, v in d["descricoes_por_bo"].items()}
        motor.emb_descricoes_por_bo = {bo_id: np.array(v) for bo_id, v in d["emb_descricoes_por_bo"].items()}
        motor.cache_scores = {
            frozenset(chave.split("||")): valor
            for chave, valor in d.get("cache_scores", {}).items()
        }
        return motor
