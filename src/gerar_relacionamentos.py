# -*- coding: utf-8 -*-
"""
Fork de gerar_relacionamentos.py (ver ../../src/gerar_relacionamentos.py).

Diferença deste fork: NÃO instancia encoder nenhum (nem SentenceTransformer
nem EncoderVLLM/vLLM) -- ver a nota dentro de main() sobre por quê. Com
vLLM no lugar do sentence_transformers original, instanciar o encoder
antes do fork() dos workers arriscaria travar/corromper os processos
filhos (contexto CUDA não é fork-safe, e vLLM já roda subprocessos
próprios como EngineCore). Como bos_conectados/score_par nunca chamam
motor.encoder (só adicionar_bo chama, e este script só consulta um
índice já pronto), a solução mais simples é não instanciar encoder algum.

Roda o caso de uso 1 do motor ("dado esse BO, quais estão conectados a
ele?") para todo BO do índice de produção e salva o resultado em JSONL:
uma linha por BO, com a lista dos BOs relacionados (score >= threshold),
o score de cada relação e o motivo.

Paralelizado com multiprocessing: bos_conectados chama o classificador
par-a-par (sem batching, ver motor_linkage.py:_feats/score_par), então
uma chamada contra o índice completo (11.838 BOs) já leva ~7s -- rodar
todos os BOs sequencialmente levaria ~24h.

O índice é carregado UMA VEZ no processo principal, e os workers são
criados por `fork` (não `spawn`), herdando esse objeto já pronto via
memória copy-on-write -- carregar o índice (JSON de ~400MB) em paralelo,
um por worker, serializa em locks de arquivo e o throughput desaba em
vez de escalar (medido na versão original com SentenceTransformer: com 4
workers carregando cada um seu próprio encoder, só 2 progrediam, os
outros ficavam presos em futex_wait).

Pré-requisito: índice de produção já construído e salvo em disco (ver
run_linkage.py, deste mesmo fork), em INDICE_PRODUCAO_PREFIXO.

Formato de saída (uma linha JSON por BO):
{
    "bo_id": "21F2058000746",
    "relacionados": [
        {"bo_id": "21F2058061842", "score": 0.83, "motivo": "mesmo(a) DOCUMENTO"},
        ...
    ]
}

LIMITAÇÃO CONHECIDA (medida numa rodada de 11.838 BOs com threshold=0.5,
default antigo de bos_conectados -- desde então corrigido para 0.9, o
valor calibrado no README seção 8): média de ~1028 relacionados/BO --
alto demais pra ser só linkage de verdade. Causa raiz: motor_linkage.py
trata LOCAL_ESTABELECIMENTO e OBJETO_VEICULO como identificador de
match exato (tipos_local/tipos_fortes), mas o NER real devolve valores
genéricos pra esses tipos ("delegacia", "casa", "residencia", "veiculo",
"motocicleta", "moto", "carro" -- não nomes próprios/placas
específicas), então BOs sem nenhuma relação real batem só por ambos
citarem "delegacia" ou "veiculo". Medido na amostra dos primeiros 2000
BOs: só 1,1% dos relacionamentos têm identificador forte de verdade
(PESSOA_NOME, DOCUMENTO, PLACA, TELEFONE, ...); 17,2% são match só de
tipo genérico; 41,8% são só "similaridade semântica" (sem motivo de
entidade). threshold=0.9 reduz o volume mas não corrige a causa raiz --
filtrar na camada de consumo (ex: descartar relacionamentos cujo motivo
cita só LOCAL_ESTABELECIMENTO/OBJETO_VEICULO/LOCAL_AREA/LOCAL_COMODO
sem nenhum identificador forte junto) ou adicionar uma lista de valores
genéricos a ignorar em
normalizar_valor_entidade antes de rodar de novo.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import warnings

# Caminhos relativos à raiz do fork (refactor/) -- rodar como:
#   cd refactor && python src/gerar_relacionamentos.py
INDICE_PRODUCAO_PREFIXO = "data/processed/indice_producao"
SAIDA_PADRAO = "data/processed/relacionamentos.jsonl"
THRESHOLD_PADRAO = 0.9  

# global do worker -- herdado do processo pai via fork(), não reconstruído.
_motor = None


def _init_worker(threads_por_worker):
    # Sem isso, cada worker abre um pool BLAS com TODOS os cores da
    # máquina -- com N workers rodando junto (mesmo só usando o motor
    # herdado, sem recarregar nada), vira oversubscription severa
    # (N x nproc threads competindo por nproc cores reais).
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads_por_worker)
    warnings.filterwarnings("ignore")


def _processar_bo(args):
    bo_id, threshold = args
    acima_threshold, _ = _motor.bos_conectados(bo_id, k=1, threshold=threshold)
    return {
        "bo_id": bo_id,
        "relacionados": [
            {"bo_id": r["bo_id"], "score": r["score"], "motivo": r["motivo"]}
            for r in acima_threshold
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indice", default=INDICE_PRODUCAO_PREFIXO,
                         help="prefixo do índice salvo (default: %(default)s)")
    parser.add_argument("--saida", default=SAIDA_PADRAO,
                         help="arquivo JSONL de saída (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_PADRAO,
                         help="score mínimo para um BO entrar como relacionado (default: %(default)s)")
    parser.add_argument("--limite", type=int, default=None,
                         help="processa só os N primeiros BOs do índice (default: todos)")
    parser.add_argument("--workers", type=int, default=8,
                         help="processos paralelos, criados por fork a partir do índice já carregado (default: %(default)s)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # Sem encoder aqui de propósito: bos_conectados/score_par (o único
    # caminho usado por este script) nunca chamam MotorDeLinkage.encoder
    # -- só adicionar_bo chama (ver motor_linkage.py). Os embeddings já
    # estão salvos no índice e MotorDeLinkage.carregar() só reidrata o
    # array, não reencoda nada. Isso importa especialmente neste fork:
    # instanciar EncoderVLLM aqui alocaria contexto CUDA + processos
    # próprios do vLLM (EngineCore) ANTES do fork() dos workers logo
    # abaixo -- CUDA não é fork-safe, arriscando travar/corromper os
    # processos filhos. Não instanciar é mais seguro que instanciar e
    # torcer para o fork não quebrar.
    print(f"Carregando índice '{args.indice}' (sem encoder -- não é necessário para bos_conectados)...")
    t0 = time.time()
    from motor_linkage import MotorDeLinkage
    from dados_reais import construir_geo_lookup
    geo_lookup = construir_geo_lookup()
    global _motor
    _motor = MotorDeLinkage.carregar(args.indice, encoder=None, geo_lookup=geo_lookup,
                                      tau_geo_km=15.0, tau_dias=30.0)
    print(f"  {len(_motor.ids)} BOs no índice ({time.time() - t0:.1f}s).")

    ids = _motor.ids[: args.limite] if args.limite else _motor.ids
    print(f"Processando {len(ids)} BOs com {args.workers} workers (fork).")

    tarefas = [(bo_id, args.threshold) for bo_id in ids]
    threads_por_worker = max(1, mp.cpu_count() // args.workers)
    print(f"  {threads_por_worker} threads BLAS por worker (evita oversubscription).")

    ctx = mp.get_context("fork")
    t0 = time.time()
    n = 0
    with ctx.Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(threads_por_worker,)) as pool, \
         open(args.saida, "w", encoding="utf-8") as f:
        for registro in pool.imap(_processar_bo, tarefas, chunksize=20):
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                dt = time.time() - t0
                taxa = n / dt
                restante = (len(ids) - n) / taxa if taxa > 0 else float("inf")
                print(f"  {n}/{len(ids)} BOs processados... "
                      f"({dt:.1f}s, ~{restante:.0f}s restantes)")

    print(f"Concluído: {n} BOs em {time.time() - t0:.1f}s. Saída: {args.saida}")


if __name__ == "__main__":
    main()
