# Motor de Linkage de BOs — v1 (handoff para backend)

> **Este é um fork** do projeto original, adaptado pra usar
> **vLLM** como encoder de texto em vez de `sentence_transformers` puro.
> `motor_linkage.py` **não muda** -- a troca é isolada num adapter
> (`encoder_vllm.py`) que expõe a mesma interface `.encode(...)`.
>
> Diferenças práticas em relação ao original:
> - **Encoder**: `EncoderVLLM` (vLLM, GPU) em vez de `SentenceTransformer`.
>   Precisa rodar `python src/converter_checkpoint_bert.py` uma vez antes
>   do primeiro uso -- vLLM não consegue tratar
>   `neuralmind/bert-base-portuguese-cased` como modelo de embedding sem
>   converter o checkpoint primeiro (ver docstring de `encoder_vllm.py`
>   e `converter_checkpoint_bert.py` pro porquê).
> - **`run_linkage.py`** (ingestão): usa `EncoderVLLM()` no lugar de
>   `SentenceTransformer(...)`.
> - **`gerar_relacionamentos.py`**: carrega o índice com `encoder=None`
>   de propósito -- `bos_conectados`/`score_par` nunca chamam o encoder
>   (só `adicionar_bo` chama), e instanciar um `LLM` do vLLM antes do
>   `fork()` dos workers arriscaria travar/corromper os processos filhos
>   (CUDA não é fork-safe).
> - **Estrutura de pastas**: padrão [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/)
>   -- `data/` segmentada por estágio (`raw/`, `external/`, `interim/`,
>   `processed/`), `models/` só com artefatos de modelo treinado, `src/`
>   só com código. Ver seção "Conteúdo do pacote" abaixo pro detalhe de
>   cada pasta.
> - **⚠️ Índice não é compatível** com o do projeto original: o pooling
>   via vLLM não é garantido bit-idêntico ao mean-pooling do
>   `sentence_transformers` (validado como semanticamente equivalente,
>   não bit-exact) -- rodar `run_linkage.py` deste fork do zero pra
>   gerar um índice próprio, não reaproveitar `indice_producao_*` do
>   projeto original.
>
> O restante deste README documenta o motor original e continua válido
> (schema de entidades, os 3 casos de uso, parâmetros calibrados) --
> só a seção de instanciação do encoder muda, ver acima.

Este pacote contém o motor de cruzamento/linkage de Boletins de Ocorrência
já treinado e validado. **Não é necessário treinar, retreinar ou rotular
nada** — o classificador já vem pronto (`.joblib`), e o backend só precisa
consumir os métodos públicos da classe `MotorDeLinkage`.

## Conteúdo do pacote

```
src/
  motor_linkage.py                   -- código-fonte da classe MotorDeLinkage (não modificado)
  encoder_vllm.py                    -- adapter: EncoderVLLM, mesma interface do SentenceTransformer
  converter_checkpoint_bert.py       -- converte um checkpoint BERT "cru" pro formato que vLLM aceita
  embed_vllm.py                      -- CLI standalone de teste (gera embedding de um texto/arquivo)
  run_linkage.py                     -- ingestão: lê data/raw/, gera data/processed/indice_producao_*
  gerar_relacionamentos.py           -- roda o caso de uso 1 em lote, gera data/processed/relacionamentos.jsonl
  dados_reais.py                     -- cruza BOs + NER pro formato esperado por adicionar_bo

data/
  raw/                                -- dados brutos, nunca modificados pelo pipeline
    bos_reconstruidos_clean_b.jsonl
    bos_reconstruidos_gemma4_e4b_preannotate_20260729_001/
  external/                           -- dados de terceiros (não gerados por este projeto)
    bertimbau-embed/                  -- checkpoint BERTimbau convertido (ver converter_checkpoint_bert.py)
  interim/                            -- dados intermediários (vazio nesta versão)
  processed/                          -- saída do pipeline: índice + relacionamentos
    indice_producao_*
    relacionamentos.jsonl

models/
  modelo_v1_classificador.joblib      -- classificador treinado (LogisticRegression), não retreinar

notebooks/
  validacao_backend.ipynb
  validacao_dados_reais.ipynb

README.md                             -- este arquivo
```

`data/`, `models/` e seus conteúdos ficam fora do controle de versão
(ver `.gitignore`) -- reproduzíveis rodando `converter_checkpoint_bert.py`
e `run_linkage.py`.

Não incluído (e não necessário): embeddings/índice da base de teste/calibração
usada durante o desenvolvimento -- o índice de produção é construído do zero,
a partir dos BOs reais, via `adicionar_bo`.

## 1. Instalação

```bash
pip install -r requirements.txt
```

`requirements.txt` (fork: vllm no lugar de sentence-transformers):
```
vllm
networkx
scikit-learn
joblib
numpy
safetensors
huggingface_hub
```

## 2. Como instanciar o motor

```python
import joblib
from encoder_vllm import EncoderVLLM
from motor_linkage import MotorDeLinkage

# 1) Encoder de texto -- roda em produção, não é só do notebook de testes.
#    Requer conversão prévia do checkpoint (uma vez só, ver seção 1 acima
#    da nota de fork no topo deste README):
#      python converter_checkpoint_bert.py
encoder = EncoderVLLM()  # default: data/bertimbau-embed (checkpoint local convertido)

# 2) Classificador já treinado -- não precisa (e não deve) ser retreinado
clf = joblib.load("modelo_v1_classificador.joblib")

# 3) Lookup de município -> (lat, lon). Se não tiver essa base ainda,
#    pode instanciar com {} -- o motor lida graciosamente com BOs sem
#    coordenada (o bloco de similaridade geográfica fica neutro).
geo_lookup = {
    "RECIFE": (-8.0476, -34.8770),
    # ...
}

motor = MotorDeLinkage(
    encoder, geo_lookup, clf=clf,
    tau_geo_km=15.0,        # decaimento da similaridade geográfica (km)
    tau_dias=30.0,          # decaimento da similaridade de data (dias)
    limiar_descricao=0.6,   # a partir de que score uma descrição conta como "batida"
)
```

**Importante:** essa é uma instância **nova** (`MotorDeLinkage(...)`), não
`MotorDeLinkage.carregar(...)`. O `.carregar()` é para recarregar um índice
de produção que **o próprio backend** já salvou anteriormente (ver seção 6),
não para reaproveitar dados da fase de calibração.

## 3. Alimentando o motor com um BO novo

```python
bo = {
    "id": "BO-2026-001234",
    "texto": "Texto completo do boletim de ocorrência...",
    "data_fato": "15/07/2026",     # formato DD/MM/AAAA
    "municipio": "RECIFE",          # precisa bater com uma chave de geo_lookup
    "natureza": "furto/roubo",      # campo estruturado, usado como fallback
}

motor.adicionar_bo(bo, entidades=entidades)  # ver seção 4 sobre `entidades`
```

Campos esperados em `bo`:

| Campo | Obrigatório | Observação |
|---|---|---|
| `id` | sim | identificador único do BO; erro se já existir no índice |
| `texto` | sim | texto completo, usado pro embedding geral |
| `data_fato` (ou `data`) | não | se ausente, o bloco de similaridade de data fica neutro |
| `municipio` | não | precisa existir em `geo_lookup`; se ausente, bloco geo fica neutro |
| `natureza` | não | usado como fallback quando a entidade `OCORRENCIA` não vem do NER |

## 4. ⚠️ Ponto crítico: de onde vêm as `entidades`

Este é o principal ponto que precisa ser decidido/confirmado antes da
implementação, porque afeta diretamente a qualidade dos 3 casos de uso.

`adicionar_bo` espera uma lista de entidades no formato:
```python
entidades = [
    {"texto": "João da Silva", "tipo": "PESSOA_NOME"},
    {"texto": "ABC-1234", "tipo": "PLACA"},
    {"texto": "furto qualificado", "tipo": "OCORRENCIA"},
    # ...
]
```

`tipo` deve ser um dos 34 valores em `SCHEMA_ENTIDADES` (dentro de
`motor_linkage.py`) -- os principais para os casos de uso são
`PESSOA_NOME`, `DOCUMENTO`, `PLACA`, `TELEFONE`, `OCORRENCIA`,
`PESSOA_DESCRICAO`, `OBJETO_VEICULO`, `OBJETO_ARMA`, entre outros.

Duas situações possíveis:

- **Se já existe um serviço de NER** que extrai essas entidades do texto do
  BO: o backend chama esse serviço antes de `adicionar_bo` e passa o
  resultado no parâmetro `entidades`.
- **Se não existe NER disponível ainda**: passando `entidades=None`, o
  motor usa um fallback interno (`extrair_entidades_por_regex`) que só
  reconhece **4 tipos** via regex simples (`PLACA`, `DOCUMENTO` via
  CPF/IMEI, `TELEFONE`). Isso é suficiente para rodar o pipeline, mas é
  **bem mais limitado** -- não captura `PESSOA_NOME` nem `OCORRENCIA`, o
  que reduz a qualidade dos casos de uso 2 e 3 (que dependem justamente
  dessas entidades).

**Confirmar com quem está entregando este pacote qual das duas situações
se aplica antes de decidir o escopo da v1.**

## 5. Os 3 casos de uso (métodos públicos)

### Caso 1 — "Dado esse BO, quais estão conectados a ele?"

```python
acima_threshold, top_k = motor.bos_conectados("BO-2026-001234", k=10, threshold=0.5)
```

- `acima_threshold`: todos os BOs com score >= threshold, ordenados por score.
- `top_k`: os `k` primeiros, independente do threshold (útil para exibir
  "os mais parecidos" mesmo que nenhum bata o threshold).

Cada item retornado:
```python
{
    "bo_id": "BO-2026-000987",
    "score": 0.83,
    "sim_texto": 0.71, "sim_geo": 0.65, "sim_data": 0.80,
    "sim_local_exato": 1.0, "sim_descricao": 0.42,
    "motivo": "mesmo(a) PLACA; descrição parecida ('arma rosa' ~ 'pistola cor-de-rosa')",
}
```

### Caso 2 — "Dado esse tipo de crime, quais grupos de BOs se conectam entre si?"

```python
grupos = motor.clusters_por_natureza("furto", threshold=0.5)
# grupos: lista de listas de bo_id, ordenada do maior grupo pro menor
```

⚠️ Este método recalcula todos os pares dentro do subconjunto filtrado
(O(n²)) -- ver seção 7 sobre performance.

### Caso 3 — "Essa pessoa tem mais de um BO?"

```python
candidatos = motor.bos_da_pessoa("João da Silva", limiar=0.85)
# [{"bo_id": "BO-...", "score": 0.94, "natureza": "furto"}, ...]
```

Usa apenas a entidade `PESSOA_NOME` (nunca `PESSOA_MENCAO`, que só indica
papel como "vítima"/"condutor", sem identificar quem é a pessoa).

## 6. Persistência (crescimento incremental em produção)

O índice cresce conforme BOs novos chegam. Para não perder o estado entre
reinícios do serviço:

```python
# ao final de um lote de ingestão, ou periodicamente:
motor.salvar("indice_producao")

# ao reiniciar o serviço:
motor = MotorDeLinkage.carregar("indice_producao", encoder, geo_lookup,
                                 tau_geo_km=15.0, tau_dias=30.0)
```

Isso gera/lê 3 arquivos: `indice_producao_classificador.joblib`,
`indice_producao_embeddings.npy`, `indice_producao_indice.json`.
O `cache_scores` (ver seção 7) também é persistido junto.

## 7. Notas de performance

- `score_par` tem cache interno (`motor.cache_scores`) -- pares já
  comparados não são recalculados. Seguro enquanto não existir edição de
  BO já inserido (hoje só existe inserção).
- Para BOs chegando um de cada vez com necessidade de resposta imediata
  na consulta seguinte, use `adicionar_bo(bo, entidades=..., calcular_scores_imediatamente=True)`
  -- pré-calcula e cacheia os scores contra todos os BOs já existentes no
  momento da inserção. Padrão é `False` (mais rápido para ingestão em
  lote; scores calculados sob demanda).
- `clusters_por_natureza` é O(n²) dentro do subconjunto filtrado por
  natureza -- adequado para consultas pontuais, não para rodar em loop
  sobre todas as naturezas a cada requisição.

## 8. Parâmetros calibrados nesta versão

| Parâmetro | Valor usado na validação |
|---|---|
| `threshold` (score de linkage) | 0.9 |
| `limiar_descricao` | 0.6 |
| `tau_geo_km` | 15.0 |
| `tau_dias` | 30.0 |
| Encoder | `neuralmind/bert-base-portuguese-cased` |
| `C` da regressão logística | 1000.0 (`class_weight="balanced"`) |

Esses valores podem ser ajustados em produção conforme feedback real de
uso, mas foram o ponto de partida validado durante a calibração.
