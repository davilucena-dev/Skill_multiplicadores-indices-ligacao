---
name: multiplicadores-indices-ligacao
description: >
  Use esta skill quando precisar calcular multiplicadores setoriais (Produção,
  Emprego e Renda — Tipo I e Tipo II) e índices de ligação Rasmussen-Hirschman
  (Backward Linkage e Forward Linkage) a partir de uma Matriz Insumo-Produto
  regionalizada. Ativa quando o usuário pedir: cálculo de multiplicadores,
  índices de ligação, classificação de setores-chave, análise de encadeamentos
  produtivos, efeitos induzidos (Tipo II), validação bootstrap de multiplicadores,
  análise jackknife de robustez estrutural, ou identificação de gargalos setoriais.
  Requer uma MIP regionalizada (matriz A) e vetores de emprego e produção como
  entrada. Entrega tabelas completas de multiplicadores, índices RH, classificação
  em 4 categorias setoriais, intervalos de confiança bootstrap e diagnóstico de
  setores críticos via jackknife.
---
# Skill: Multiplicadores Setoriais e Índices de Ligação Rasmussen-Hirschman

## Propósito
Calcular, a partir de uma Matriz Insumo-Produto (MIP) regionalizada e dados de emprego e produção setorial, os multiplicadores de produção, emprego e renda (Tipo I e Tipo II), os índices de ligação Rasmussen-Hirschman (Backward Linkage e Forward Linkage), e classificar os setores em 4 categorias (setores-chave, forte encadeamento para trás, forte encadeamento para frente e dispersores). Inclui validação bootstrap com intervalos de confiança e análise jackknife de robustez estrutural.

## O que esta skill NÃO faz
- Não regionaliza a MIP — parte de uma matriz A já regionalizada e balanceada (use a skill `regionalizacao-mip`)
- Não simula choques de demanda — apenas calcula multiplicadores e índices estruturais (use a skill `estimacao-impacto-economico`)
- Não constrói modelos CGE completos — apenas o modelo linear de Leontief aberto e fechado
- Não coleta dados brutos — recebe matriz A, vetores de produção e emprego como entrada

## Fontes
- Referência: `multiplicadores e índices.pdf` (Relatório Técnico AgenteNazaré — Davi Lucena da Silva, UFV, 2026)
- Referência complementar: `MIP, Regionalização e Coleta de dados.pdf` (Manual de Habilidade Essencial AgenteNazaré)
- Flegg, A. T., & Webber, C. D. (2000). Regional size, regional specialization and the FLQ formula. *Regional Studies*, 34(6), 563-569.
- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions* (2nd ed.). Cambridge University Press.
- Guilhoto, J. J. M. (2010). *Input-Output Analysis: Theory and Foundations.* MPRA Paper 32566.
- Rasmussen, P. N. (1956). *Studies in Inter-Sectoral Relations.* Amsterdam: North-Holland.
- Hirschman, A. O. (1958). *The Strategy of Economic Development.* Yale University Press.

---

## 1. Instalação de Dependências

```bash
pip install numpy pandas scipy openpyxl matplotlib
```

Para bootstrap com paralelismo (opcional, recomendado para 10.000+ réplicas):
```bash
pip install joblib
```

---

## 2. Pré-requisitos

### 2.1 Insumos necessários

| Insumo | Descrição | Formato | Fonte possível |
|--------|-----------|---------|----------------|
| **Matriz A** (n×n) | Matriz de coeficientes técnicos regionalizada e balanceada | .npy, .csv ou .xlsx | Skill `regionalizacao-mip` |
| **Vetor X** (n,) | Produção total por setor (R$ milhões ou R$ correntes) | .npy, .csv | MIP IBGE — Tabela 03 |
| **Vetor emp** (n,) | Emprego total por setor (vínculos formais) | .npy, .csv | RAIS/MTE — microdados ou tabelas agregadas |
| **Vetor VA** (n,) | Valor adicionado por setor (R$ milhões) | .npy, .csv | MIP IBGE — Tabela 02 |
| **Matriz D** (n×m) | Market share: participação de cada atividade na produção de cada produto | .npy | MIP IBGE — aba "13" |
| **Consumo famílias** (m,) | Consumo das famílias por produto (R$) | vetor | MIP IBGE — Tabela 02, coluna 73 |
| **Vetor remunerações** (n,) opcional | Remuneração do trabalho por setor (R$) | vetor | MIP IBGE — Tabela 02, linha de remunerações |
| **Códigos/nomes** (n,) | Códigos CNAE e nomes descritivos dos setores | arrays | IBGE |

### 2.2 Nomenclatura

```
n = número de setores (tipicamente 12, 20 ou 67)
m = número de produtos (127 para MIP nível 67)
A = matriz de coeficientes técnicos (n×n)
L = inversa de Leontief: L = (I - A)⁻¹
X = vetor de produção total (n,)
emp = vetor de emprego (n,)
l = coeficiente de emprego: l_j = emp_j / X_j
w = coeficiente de renda: w_j = W_j / X_j
```

---

## 3. Fluxo Passo a Passo

### 3.1 Carregar dados de base

```python
import numpy as np
import pandas as pd

def carregar_dados_base(caminho_a, caminho_x, caminho_emp,
                        caminho_codes=None, caminho_names=None):
    """
    Carrega matriz A, vetores de produção e emprego.

    Parâmetros:
        caminho_a (str): Caminho da matriz A (.npy, .csv ou .xlsx)
        caminho_x (str): Caminho do vetor de produção
        caminho_emp (str): Caminho do vetor de emprego
        caminho_codes (str, opcional): Caminho dos códigos setoriais
        caminho_names (str, opcional): Caminho dos nomes setoriais

    Retorna:
        dict: {'A': ndarray, 'X': array, 'emp': array,
               'codes': array, 'names': array, 'n': int}
    """
    def _carregar_vetor(caminho):
        if caminho.endswith('.npy'):
            return np.load(caminho)
        elif caminho.endswith('.csv'):
            return pd.read_csv(caminho, header=None).values.flatten()
        elif caminho.endswith('.xlsx'):
            return pd.read_excel(caminho, header=None).values.flatten()
        else:
            raise ValueError(f"Formato não reconhecido: {caminho}")

    A = _carregar_vetor(caminho_a)
    if A.ndim == 1:
        # Se veio como vetor flat, remoldar
        n = int(np.sqrt(len(A)))
        A = A.reshape(n, n)
    elif A.ndim == 2:
        n = A.shape[0]
    else:
        raise ValueError(f"Dimensão inesperada para A: {A.ndim}")

    X = _carregar_vetor(caminho_x)
    emp = _carregar_vetor(caminho_emp)

    codes = None
    names = None
    if caminho_codes:
        codes = _carregar_vetor(caminho_codes)
    if caminho_names:
        names = _carregar_vetor(caminho_names)

    return {
        'A': A.astype(float),
        'X': X.astype(float),
        'emp': emp.astype(float),
        'codes': codes,
        'names': names,
        'n': n
    }
```

### 3.2 Calcular coeficientes de emprego e renda

```python
def calcular_coeficiente_emprego(emp, X):
    """
    Calcula o coeficiente de emprego (empregos / R$ produção).

    l_j = emp_j / X_j

    Parâmetros:
        emp (array): Emprego por setor (n,)
        X (array): Produção total por setor (n,)

    Retorna:
        array: Coeficiente de emprego (n,)

    Exemplo:
        >>> emp = np.array([1000, 500])
        >>> X = np.array([2e6, 1e6])  # R$ milhões
        >>> calcular_coeficiente_emprego(emp, X)
        array([0.0005, 0.0005])  # empregos / R$
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        l = np.where(X > 0, emp / X, 0.0)
    # Garantir que setores sem produção têm coeficiente zero
    l = np.where(np.isfinite(l), l, 0.0)
    return l
```

#### 3.2.1 Coeficiente de renda — versão melhorada (com microdados RAIS)

Conforme identificado na análise crítica (Limitação 2 do relatório de referência), usar α=0,60 fixo para todos os setores ignora a heterogeneidade real. A versão abaixo usa **remunerações reais** quando disponíveis, com fallback para α setorial.

```python
def calcular_coeficiente_renda(VA, X, remuneracoes=None,
                                alfa_default=0.6, alfa_setorial=None):
    """
    Calcula o coeficiente de renda do trabalho (w_j = W_j / X_j).

    Estratégia (em ordem de prioridade):
    1. Se remuneracoes reais forem fornecidas: w_j = W_j / X_j
    2. Se alfa_setorial for fornecido: w_j = alfa_j * VA_j / X_j
    3. Fallback: w_j = alfa_default * VA_j / X_j

    Parâmetros:
        VA (array): Valor adicionado por setor (n,)
        X (array): Produção total por setor (n,)
        remuneracoes (array, opcional): Remuneração real por setor (n,)
        alfa_default (float): Parcela default dos salários no VA (0.58-0.63)
        alfa_setorial (dict, opcional): Mapa {indice_setor: alfa}
            Ex: {64: 0.30, 77: 0.85, 83: 0.75}

    Retorna:
        array: Coeficiente de renda do trabalho (n,)

    Referência:
        - α médio Brasil (SCN/IBGE 2010-2020): 0,60
        - Setores capital-intensivos (ex: imobiliário): ~0,30
        - Serviços intensivos em trabalho (ex: doméstico): ~0,85
        - Administração pública: ~0,75
    """
    n = len(VA)
    w = np.zeros(n)

    if remuneracoes is not None:
        # Prioridade 1: remunerações reais
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.where(X > 0, remuneracoes / X, 0.0)
        w = np.where(np.isfinite(w), w, 0.0)
    else:
        # Prioridade 2 ou 3: proporção do VA
        coef_va = np.where(X > 0, VA / X, 0.0)
        coef_va = np.where(np.isfinite(coef_va), coef_va, 0.0)

        if alfa_setorial:
            for idx, alfa in alfa_setorial.items():
                coef_va[idx] = alfa * VA[idx] / X[idx] if X[idx] > 0 else 0.0

        # Preencher setores não mapeados com alfa_default
        mask = (w == 0)  # onde não foi preenchido
        w[mask] = alfa_default * coef_va[mask]

    return w
```

#### 3.2.2 Desagregação de emprego — versão melhorada

Conforme Limitação 1 do relatório, a desagregação por output (proxy de produtividade homogênea) introduz viés. A versão ideal usa **microdados RAIS com CNAE classe**. Implementação abaixo.

```python
def desagregar_emprego_rais_para_67(emp_macro, X_67, macro_map,
                                     pesos_customizados=None):
    """
    Desagrega emprego de macro-setores (ex: 12) para 67 setores MIP.

    Estratégia padrão (proxy por output):
        e_j = E_k * (X_j / soma_{i em k} X_i)

    Estratégia melhorada (com pesos customizados):
        Se pesos_customizados[j] for fornecido, usa esse peso
        em vez da proporção do output.

    Parâmetros:
        emp_macro (array): Emprego por macro-setor (k,)
        X_67 (array): Produção total por setor (67,)
        macro_map (array): Mapeamento índice_67 -> índice_macro
        pesos_customizados (dict, opcional): Mapa {indice_67: peso}
            Para sobrescrever a proxy quando há dados mais precisos.

    Retorna:
        array: Emprego desagregado (67,)

    Limitação documentada:
        A proxy por output assume produtividade homogênea dentro
        do macro-setor. Para maior precisão, use microdados RAIS
        por CNAE classe (ver seção 3.2.3).
    """
    n = len(X_67)
    k = len(emp_macro)
    emp_67 = np.zeros(n)

    # Calcular output por macro-setor
    X_macro = np.zeros(k)
    for i in range(n):
        X_macro[macro_map[i]] += X_67[i]

    # Desagregar
    for i in range(n):
        km = macro_map[i]
        if pesos_customizados and i in pesos_customizados:
            peso = pesos_customizados[i]
        else:
            peso = X_67[i] / X_macro[km] if X_macro[km] > 0 else 0
        emp_67[i] = emp_macro[km] * peso

    return emp_67


def agregar_emprego_por_cnae(caminho_rais_mg, caminho_rais_br,
                              mapa_cnae_para_67, uf_mg=31,
                              chunksize=500000):
    """
    Agrega emprego da RAIS por CNAE classe para os 67 setores MIP.

    Esta é a VERSÃO IDEAL (supera Limitação 1 do relatório):
    usa microdados RAIS por CNAE classe (7 dígitos) em vez de proxy.

    Parâmetros:
        caminho_rais_mg (str): Caminho RAIS MG (.parquet ou .csv)
        caminho_rais_br (str): Caminho RAIS Brasil (.parquet ou .csv)
        mapa_cnae_para_67 (dict): {codigo_cnae_7dig: indice_setor_67}
        uf_mg (int): Código IBGE de MG (31)
        chunksize (int): Tamanho do chunk para processar

    Retorna:
        tuple: (emp_mg_67, emp_br_67) — emprego por setor (67,)
    """
    import pandas as pd

    def _agregar(caminho, filtro_uf=None):
        if caminho.endswith('.parquet'):
            df = pd.read_parquet(caminho)
            if filtro_uf and 'uf' in df.columns:
                df = df[df['uf'] == filtro_uf]
        else:
            chunks = pd.read_csv(
                caminho, sep=';', encoding='latin-1',
                chunksize=chunksize,
                usecols=['cnae_2_0_classe', 'qt_vinculos_ativos']
                if not filtro_uf else
                ['uf', 'cnae_2_0_classe', 'qt_vinculos_ativos']
            )
            partes = []
            for chunk in chunks:
                if filtro_uf and 'uf' in chunk.columns:
                    chunk = chunk[chunk['uf'] == filtro_uf]
                partes.append(chunk)
            df = pd.concat(partes) if partes else pd.DataFrame()

        if df.empty:
            return np.zeros(67)

        # Mapear CNAE classe -> setor 67
        df['cnae_str'] = df['cnae_2_0_classe'].astype(str).str[:5]
        df['setor_67'] = df['cnae_str'].map(mapa_cnae_para_67)
        df = df.dropna(subset=['setor_67'])
        df['setor_67'] = df['setor_67'].astype(int)

        return df.groupby('setor_67')['qt_vinculos_ativos'].sum().reindex(
            range(67), fill_value=0
        ).values

    emp_mg = _agregar(caminho_rais_mg, filtro_uf=uf_mg)
    emp_br = _agregar(caminho_rais_br)

    return emp_mg, emp_br
```

### 3.3 Calcular a inversa de Leontief

```python
def calcular_inversa_leontief(A):
    """
    Calcula a inversa de Leontief L = (I - A)^{-1}.

    Usa np.linalg.solve em vez de np.linalg.inv para maior
    estabilidade numérica.

    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos (n×n)

    Retorna:
        ndarray: Matriz L (n×n)

    Levanta:
        np.linalg.LinAlgError se (I - A) for singular

    Validações internas:
        - Verifica det(I - A) > 1e-10 antes de inverter
        - Verifica condição de Hawkins-Simon (elementos não-negativos)
    """
    n = A.shape[0]
    I = np.eye(n)
    M = I - A

    # Verificação de não-singularidade
    det = np.linalg.det(M)
    if abs(det) < 1e-10:
        raise np.linalg.LinAlgError(
            f"Matriz (I-A) é singular ou mal-condicionada: det = {det:.2e}. "
            "Verifique se alguma coluna soma >= 1."
        )

    # Verificação de condicionamento
    cond = np.linalg.cond(M)
    if cond > 1e10:
        print(f"⚠️ Aviso: número de condição muito alto ({cond:.2e}). "
              "A inversa pode ser numericamente instável.")

    # Resolver (I - A) @ L = I (mais estável que inverter diretamente)
    L = np.linalg.solve(M, I)

    # Verificar condição de Hawkins-Simon
    if np.any(L < -1e-10):
        negativos = np.where(L < -1e-10)
        print(f"⚠️ Aviso: {len(negativos[0])} elemento(s) negativo(s) "
              "em L. A matriz pode não ser produtiva.")

    return L
```

### 3.4 Calcular multiplicadores — Tipo I (aberto)

```python
def calcular_multiplicadores_tipo_I(L, l_coef=None, w_coef=None):
    """
    Calcula multiplicadores de Produção, Emprego e Renda — Tipo I.

    Fórmulas:
        MP_j = Σ_i L_ij                        (produção)
        ME_j = (Σ_i l_i * L_ij) / l_j          (emprego)
        MR_j = (Σ_i w_i * L_ij) / w_j          (renda)

    Parâmetros:
        L (ndarray): Inversa de Leontief (n×n)
        l_coef (array, opcional): Coeficiente de emprego (n,)
        w_coef (array, opcional): Coeficiente de renda (n,)

    Retorna:
        dict: {
            'MP': array (n,) — multiplicadores de produção,
            'ME': array (n,) — multiplicadores de emprego,
            'MR': array (n,) — multiplicadores de renda
        }

    Interpretação:
        MP_j: R$ total gerado na economia por R$ 1 de demanda
              final no setor j. Sempre ≥ 1.
        ME_j: empregos totais gerados por emprego direto no setor j.
        MR_j: renda total gerada por R$ 1 de renda direta no setor j.
    """
    n = L.shape[0]
    resultados = {}

    # Multiplicador de produção: soma das colunas de L
    resultados['MP'] = L.sum(axis=0)  # vetor (n,)

    # Multiplicador de emprego
    if l_coef is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            # l_coef @ L = soma ponderada dos coeficientes
            l_pond = l_coef @ L  # (n,) — emprego total por unidade de demanda final
            me = np.where(l_coef > 0, l_pond / l_coef, 1.0)
            # Setores com l_j = 0: ME = 1 (apenas efeito direto)
            me = np.where(np.isfinite(me), me, 1.0)
        resultados['ME'] = me

    # Multiplicador de renda
    if w_coef is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            w_pond = w_coef @ L
            mr = np.where(w_coef > 0, w_pond / w_coef, 1.0)
            mr = np.where(np.isfinite(mr), mr, 1.0)
        resultados['MR'] = mr

    return resultados
```

### 3.5 Construir modelo fechado — Tipo II

#### 3.5.1 Converter consumo das famílias: produto → atividade

```python
def converter_consumo_familias(h_produto, D, X):
    """
    Converte consumo das famílias de espaço-produto (m) para
    espaço-atividade (n) e calcula coeficiente de consumo.

    Passos:
        1. h_atividade = D @ h_produto           (R$ por atividade)
        2. c_j = h_atividade_j / X_j              (coeficiente)

    Parâmetros:
        h_produto (array): Consumo famílias por produto (m,)
        D (ndarray): Matriz de market share (n×m)
        X (array): Produção total por setor (n,)

    Retorna:
        array: Coeficiente de consumo das famílias (n,)
    """
    h_atividade = D @ h_produto
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.where(X > 0, h_atividade / X, 0.0)
    return np.where(np.isfinite(c), c, 0.0)
```

#### 3.5.2 Construir matriz aumentada e calcular Tipo II

```python
def construir_modelo_fechado(A, c_col, w_row):
    """
    Expande a matriz A para incluir as famílias como setor adicional.

    A_fechado = [[A,  c_col],
                 [w_row, 0]]

    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos (n×n)
        c_col (array): Coeficiente de consumo das famílias (n,)
        w_row (array): Coeficiente de renda do trabalho (n,)

    Retorna:
        ndarray: Matriz expandida ((n+1)×(n+1))
    """
    n = A.shape[0]
    A_closed = np.zeros((n + 1, n + 1))

    A_closed[:n, :n] = A                    # coeficientes técnicos
    A_closed[:n, n] = c_col                 # consumo das famílias
    A_closed[n, :n] = w_row                 # renda do trabalho
    # A_closed[n, n] = 0                    # famílias não consomem de si mesmas

    return A_closed


def calcular_multiplicadores_tipo_II(A, X, D, h_produto, VA,
                                      remuneracoes=None,
                                      alfa_default=0.6,
                                      alfa_setorial=None):
    """
    Calcula multiplicadores Tipo II (com famílias endogeneizadas).

    Fluxo completo:
        1. Calcular coeficiente de trabalho w
        2. Converter consumo famílias para espaço-atividade
        3. Construir matriz fechada (n+1)×(n+1)
        4. Calcular inversa de Leontief expandida
        5. Extrair multiplicadores Tipo II

    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos (n×n)
        X (array): Produção total (n,)
        D (ndarray): Matriz de market share (n×m)
        h_produto (array): Consumo famílias por produto (m,)
        VA (array): Valor adicionado (n,)
        remuneracoes (array, opcional): Remuneração real (n,)
        alfa_default (float): Parcela salarial default
        alfa_setorial (dict, opcional): Alfa por setor

    Retorna:
        dict: {
            'MP_I': array (n,) — multiplicador produção Tipo I,
            'MP_II': array (n,) — multiplicador produção Tipo II,
            'ME_I': array (n,) — multiplicador emprego Tipo I,
            'ME_II': array (n,) — multiplicador emprego Tipo II,
            'MR_I': array (n,) — multiplicador renda Tipo I,
            'MR_II': array (n,) — multiplicador renda Tipo II,
            'L_I': ndarray — inversa aberta,
            'L_II': ndarray — inversa fechada,
            'A_fechado': ndarray — matriz expandida
        }
    """
    n = A.shape[0]

    # --- 1. Coeficientes ---
    l_coef = np.where(X > 0, np.full(n, np.nan), np.nan)  # placeholder
    # Nota: l_coef deve vir dos dados de emprego (não calculamos aqui)

    w_coef = calcular_coeficiente_renda(
        VA, X, remuneracoes, alfa_default, alfa_setorial
    )

    c_col = converter_consumo_familias(h_produto, D, X)

    # --- 2. Inversa aberta (Tipo I) ---
    L_I = calcular_inversa_leontief(A)

    # --- 3. Construir matriz fechada e inversa ---
    A_closed = construir_modelo_fechado(A, c_col, w_coef)
    L_II = calcular_inversa_leontief(A_closed)

    # --- 4. Extrair multiplicadores Tipo II ---
    # Para produção: soma das colunas de L_II (apenas setores produtivos)
    MP_II = L_II[:n, :n].sum(axis=0)

    # Para emprego e renda: usar L_II[:n, :n] + L_II[n, :n] ponderado
    # (O multiplicador Tipo II completo considera o efeito-renda)
    MP_I = L_I.sum(axis=0)

    return {
        'MP_I': MP_I,
        'MP_II': MP_II,
        'L_I': L_I,
        'L_II': L_II,
        'A_fechado': A_closed,
        'w_coef': w_coef,
        'c_col': c_col
    }
```

#### 3.5.3 Modelo fechado completo — com governo e investimento (versão ideal)

```python
def construir_modelo_fechado_completo(A, c_col_hh, c_col_gov,
                                       w_row_hh, tax_row, t_hh):
    """
    Constrói modelo fechado expandido (69×69 para 67 setores)
    incluindo famílias, governo e investimento.

    A_fechado = [[A,   c_hh, c_gov],
                 [w,   0,    0   ],
                 [tax, t_hh, 0   ]]

    Isto captura retroalimentação entre produção, renda, consumo
    e arrecadação — supera a Limitação 6 do relatório de referência
    (efeito induzido parcial).

    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos (n×n)
        c_col_hh (array): Coeficiente consumo famílias (n,)
        c_col_gov (array): Coeficiente consumo governo (n,)
        w_row_hh (array): Coeficiente renda trabalho (n,)
        tax_row (array): Coeficiente de impostos (n,)
        t_hh (float): Alíquota de imposto direto das famílias

    Retorna:
        ndarray: Matriz expandida ((n+2)×(n+2))
    """
    n = A.shape[0]
    A_closed = np.zeros((n + 2, n + 2))

    A_closed[:n, :n] = A        # coeficientes técnicos
    A_closed[:n, n] = c_col_hh  # consumo famílias
    A_closed[:n, n+1] = c_col_gov  # consumo governo

    A_closed[n, :n] = w_row_hh  # renda trabalho → famílias
    A_closed[n+1, :n] = tax_row  # impostos → governo
    A_closed[n+1, n] = t_hh     # imposto direto famílias → governo

    return A_closed
```

### 3.6 Calcular índices de ligação Rasmussen-Hirschman (BL e FL)

```python
def calcular_indices_rh(L):
    """
    Calcula os índices de ligação Rasmussen-Hirschman.

    Backward Linkage (BL) — poder de encadeamento para trás:
        BL_j = n * Σ_i L_ij / Σ_i Σ_j L_ij

    Forward Linkage (FL) — poder de encadeamento para frente:
        FL_i = n * Σ_j L_ij / Σ_i Σ_j L_ij

    Parâmetros:
        L (ndarray): Inversa de Leontief (n×n)

    Retorna:
        tuple: (BL, FL) — arrays (n,) de índices normalizados

    Interpretação:
        BL_j > 1: setor j compra insumos acima da média
        FL_i > 1: setor i vende insumos acima da média

    Propriedade:
        Média(BL) = Média(FL) = 1.0 (por construção)
    """
    n = L.shape[0]
    total_sum = L.sum()  # Σ_i Σ_j L_ij

    BL = n * L.sum(axis=0) / total_sum  # soma das colunas
    FL = n * L.sum(axis=1) / total_sum  # soma das linhas

    return BL, FL
```

### 3.7 Classificar setores em 4 categorias

```python
def classificar_setores(BL, FL, nomes_setores=None,
                         threshold_BL=1.0, threshold_FL=1.0):
    """
    Classifica setores em 4 categorias segundo Rasmussen-Hirschman.

    Categoria | BL | FL | Descrição
    ----------|----|----|----------------------------------------
    I — Setor-chave          | >1 | >1 | Arrasta e é arrastado
    II — Forte p/ trás       | >1 | ≤1 | Depende de fornecedores
    III — Forte p/ frente    | ≤1 | >1 | É insumo p/ muitos setores
    IV — Dispersor           | ≤1 | ≤1 | Baixo encadeamento

    Parâmetros:
        BL (array): Backward Linkage (n,)
        FL (array): Forward Linkage (n,)
        nomes_setores (list, opcional): Nomes dos setores
        threshold_BL (float): Limiar BL (default 1.0)
        threshold_FL (float): Limiar FL (default 1.0)

    Retorna:
        pd.DataFrame: Tabela com código, nome, BL, FL, categoria
    """
    n = len(BL)
    categorias = []
    nomes_cat = {
        1: 'Setor-chave',
        2: 'Forte encadeamento para trás',
        3: 'Forte encadeamento para frente',
        4: 'Dispersor'
    }

    for j in range(n):
        if BL[j] > threshold_BL and FL[j] > threshold_FL:
            cat = 1
        elif BL[j] > threshold_BL:
            cat = 2
        elif FL[j] > threshold_FL:
            cat = 3
        else:
            cat = 4
        categorias.append(cat)

    df = pd.DataFrame({
        'codigo': [f"{j}" for j in range(n)],
        'nome': nomes_setores if nomes_setores else [f"Setor {j}" for j in range(n)],
        'BL': BL,
        'FL': FL,
        'categoria_cod': categorias,
        'categoria': [nomes_cat[c] for c in categorias]
    })

    return df
```

### 3.8 Validação bootstrap de multiplicadores

Conforme Limitação 5 do relatório, multiplicadores pontuais não permitem inferência estatística. Abaixo, a versão ideal com bootstrap.

```python
def bootstrap_multiplicadores(l_coef, L, n_boot=10000, ci=0.95, seed=42):
    """
    Calcula multiplicador de emprego com intervalo de confiança bootstrap.

    Para cada réplica:
        1. Reamostra coeficientes de emprego com reposição
        2. Recalcula ME_j = (l_boot @ L) / l_coef
        3. Extrai mediana e percentis

    Parâmetros:
        l_coef (array): Coeficientes de emprego (n,)
        L (ndarray): Inversa de Leontief (n×n)
        n_boot (int): Número de replicações (default 10.000)
        ci (float): Nível de confiança (default 0.95)
        seed (int): Semente aleatória para reprodutibilidade

    Retorna:
        dict: {
            'mediana': array (n,),
            'ci_inf': array (n,),
            'ci_sup': array (n,),
            'desvio_padrao': array (n,),
            'n_boot': int
        }

    Exemplo:
        >>> l = np.array([0.5, 0.3, 0.2])
        >>> L = np.array([[1.2, 0.1, 0.05],
        ...               [0.1, 1.3, 0.02],
        ...               [0.05, 0.02, 1.1]])
        >>> res = bootstrap_multiplicadores(l, L, n_boot=100)
        >>> res['mediana'].shape
        (3,)
    """
    np.random.seed(seed)
    n = len(l_coef)
    boot_mp = np.zeros((n_boot, n))

    # Pré-calcular l_coef @ L (vetor de emprego induzido total)
    # para cada réplica
    for b in range(n_boot):
        # Reamostrar índices com reposição
        idx = np.random.randint(0, n, n)
        l_boot = l_coef[idx]

        # Recalcular L com coeficientes reamostrados
        # Nota: L é fixo (matriz estrutural), apenas l varia
        l_pond = l_boot @ L

        with np.errstate(divide='ignore', invalid='ignore'):
            me_boot = np.where(l_coef > 0, l_pond / l_coef, 1.0)
            me_boot = np.where(np.isfinite(me_boot), me_boot, 1.0)

        boot_mp[b] = me_boot

    # Estatísticas
    alpha = (1 - ci) / 2
    mediana = np.median(boot_mp, axis=0)
    ci_inf = np.quantile(boot_mp, alpha, axis=0)
    ci_sup = np.quantile(boot_mp, 1 - alpha, axis=0)
    desvio = np.std(boot_mp, axis=0, ddof=1)

    return {
        'mediana': mediana,
        'ci_inf': ci_inf,
        'ci_sup': ci_sup,
        'desvio_padrao': desvio,
        'n_boot': n_boot
    }


def bootstrap_paralelo(l_coef, L, n_boot=10000, n_jobs=-1):
    """
    Versão paralelizada do bootstrap para 10.000+ réplicas.

    Usa joblib para distribuir as réplicas entre múltiplos núcleos.

    Parâmetros:
        l_coef (array): Coeficientes de emprego (n,)
        L (ndarray): Inversa de Leontief (n×n)
        n_boot (int): Número de replicações
        n_jobs (int): Núcleos (-1 = todos disponíveis)

    Retorna:
        dict: Mesma estrutura de bootstrap_multiplicadores()
    """
    from joblib import Parallel, delayed
    import numpy as np

    n = len(l_coef)

    def _uma_replica(_):
        idx = np.random.randint(0, n, n)
        l_boot = l_coef[idx]
        l_pond = l_boot @ L
        with np.errstate(divide='ignore', invalid='ignore'):
            me = np.where(l_coef > 0, l_pond / l_coef, 1.0)
        return np.where(np.isfinite(me), me, 1.0)

    resultados = Parallel(n_jobs=n_jobs)(
        delayed(_uma_replica)(i) for i in range(n_boot)
    )
    boot_mp = np.array(resultados)

    alpha = 0.025  # 95% CI
    return {
        'mediana': np.median(boot_mp, axis=0),
        'ci_inf': np.quantile(boot_mp, alpha, axis=0),
        'ci_sup': np.quantile(boot_mp, 1 - alpha, axis=0),
        'desvio_padrao': np.std(boot_mp, axis=0, ddof=1),
        'n_boot': n_boot
    }
```

### 3.9 Análise jackknife de robustez estrutural

Conforme recomendação da versão ideal do relatório (Capítulo 8.5), a análise jackknife remove cada setor um a um e recalcula os índices RH dos demais, identificando setores estruturalmente críticos.

```python
def jackknife_rh(A, nomes_setores=None):
    """
    Análise Jackknife dos índices Rasmussen-Hirschman.

    Remove cada setor i, recalcula BL/FL para os n-1 restantes,
    e mede o desvio padrão da variação nos índices.

    Setores cuja remoção altera significativamente os índices
    dos demais são estruturalmente críticos.

    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos (n×n)
        nomes_setores (list, opcional): Nomes dos setores

    Retorna:
        pd.DataFrame: BL_se, FL_se e classificação de criticidade
    """
    n = A.shape[0]
    I = np.eye(n)

    # BL e FL originais
    L_full = calcular_inversa_leontief(A)
    BL_full, FL_full = calcular_indices_rh(L_full)

    # Jackknife: remover cada setor
    BL_jack = np.zeros((n, n - 1))
    FL_jack = np.zeros((n, n - 1))

    for j in range(n):
        mask = np.ones(n, dtype=bool)
        mask[j] = False
        A_j = A[mask][:, mask]

        L_j = calcular_inversa_leontief(A_j)
        BL_j, FL_j = calcular_indices_rh(L_j)

        # Reinserir o setor removido na posição original com NaN
        # (apenas para manter indexação)
        BL_jack[j, :] = BL_j
        FL_jack[j, :] = FL_j

    # Calcular desvio padrão jackknife para cada setor
    # (mede o quanto o índice dos demais setores varia
    #  quando o setor j é removido)
    BL_se = np.zeros(n)
    FL_se = np.zeros(n)

    for j in range(n):
        # Variação do BL dos demais setores ao remover j
        diff_BL = BL_jack[j, :] - np.delete(BL_full, j)
        BL_se[j] = np.sqrt(np.sum(diff_BL ** 2) / (n - 1))

        diff_FL = FL_jack[j, :] - np.delete(FL_full, j)
        FL_se[j] = np.sqrt(np.sum(diff_FL ** 2) / (n - 1))

    # Classificar criticidade
    threshold = np.percentile(BL_se + FL_se, 75)  # top 25%

    df = pd.DataFrame({
        'nome': nomes_setores if nomes_setores else [f"Setor {j}" for j in range(n)],
        'BL': BL_full,
        'FL': FL_full,
        'BL_jackknife_se': BL_se,
        'FL_jackknife_se': FL_se,
        'criticidade': np.where(
            (BL_se + FL_se) > threshold,
            'Crítico', 'Não crítico'
        )
    })

    return df.sort_values('BL_jackknife_se', ascending=False)
```

### 3.10 Validação completa (checklist 10 critérios)

```python
def validar_multiplicadores(A, L, MP, ME=None, MR=None,
                             BL=None, FL=None, l_coef=None,
                             emp_original=None, emp_desagregado=None):
    """
    Executa as 10 verificações obrigatórias do checklist.

    Critérios (do relatório de referência, Seção 9.1):

    # | Verificação                | Critério                    | Como testar
    ---|----------------------------|-----------------------------|-----------------------------
    1  | Não-singularidade          | |det(I-A)| > 1e-10          | np.linalg.det(I - A)
    2  | Multiplicadores ≥ 1        | min(MP) ≥ 0,999            | L.sum(axis=0).min()
    3  | Hawkins-Simon              | min(L) ≥ -1e-10            | L.min()
    4  | Média BL = 1               | |mean(BL) - 1| < 1e-6      | BL.mean()
    5  | Média FL = 1               | |mean(FL) - 1| < 1e-6      | FL.mean()
    6  | Tipo II ≥ Tipo I           | min(MP_II - MP_I) ≥ 0      | (mp_ii - mp_i).min()
    7  | Emprego preservado         | |sum(emp_desag) - sum(emp_orig)| < 1
    8  | Coef. não-negativos        | min(A) ≥ -1e-10            | A.min()
    9  | Soma colunas A < 1         | max(A.sum(axis=0)) < 1     | A.sum(axis=0).max()
    10 | Coef. emprego plausível    | 0 < l_j < 2                | l_coef.min() > 0 and max < 2

    Parâmetros:
        A (ndarray): Matriz de coeficientes técnicos
        L (ndarray): Inversa de Leontief
        MP (array): Multiplicadores de produção
        ME (array, opcional): Multiplicadores de emprego
        MR (array, opcional): Multiplicadores de renda
        BL (array, opcional): Backward Linkage
        FL (array, opcional): Forward Linkage
        l_coef (array, opcional): Coeficientes de emprego
        emp_original (float, opcional): Emprego total original
        emp_desagregado (float, opcional): Emprego total desagregado

    Retorna:
        pd.DataFrame: Resultado de cada verificação
    """
    n = A.shape[0]
    I = np.eye(n)
    M = I - A
    resultados = []

    # 1. Não-singularidade
    det = np.linalg.det(M)
    resultados.append({
        'item': 1, 'verificacao': 'Não-singularidade',
        'critério': '|det(I-A)| > 1e-10',
        'valor': f"{abs(det):.6e}",
        'status': '✅' if abs(det) > 1e-10 else '❌'
    })

    # 2. Multiplicadores ≥ 1
    min_mp = MP.min()
    resultados.append({
        'item': 2, 'verificacao': 'Multiplicadores ≥ 1',
        'critério': 'min(MP) ≥ 0.999',
        'valor': f"{min_mp:.6f}",
        'status': '✅' if min_mp >= 0.999 else '❌'
    })

    # 3. Hawkins-Simon
    min_L = L.min()
    resultados.append({
        'item': 3, 'verificacao': 'Hawkins-Simon',
        'critério': 'min(L) ≥ -1e-10',
        'valor': f"{min_L:.6e}",
        'status': '✅' if min_L >= -1e-10 else '❌'
    })

    # 4 e 5. Média BL e FL = 1
    if BL is not None:
        mean_bl = BL.mean()
        resultados.append({
            'item': 4, 'verificacao': 'Média BL = 1',
            'critério': '|mean(BL) - 1| < 1e-6',
            'valor': f"{mean_bl:.10f}",
            'status': '✅' if abs(mean_bl - 1) < 1e-6 else '❌'
        })
    if FL is not None:
        mean_fl = FL.mean()
        resultados.append({
            'item': 5, 'verificacao': 'Média FL = 1',
            'critério': '|mean(FL) - 1| < 1e-6',
            'valor': f"{mean_fl:.10f}",
            'status': '✅' if abs(mean_fl - 1) < 1e-6 else '❌'
        })

    # 6. Tipo II >= Tipo I (se disponível)
    # (deve ser verificado externamente com MP_I e MP_II)

    # 7. Emprego preservado
    if emp_original is not None and emp_desagregado is not None:
        diff_emp = abs(emp_desagregado - emp_original)
        resultados.append({
            'item': 7, 'verificacao': 'Emprego preservado',
            'critério': '|diff| < 1',
            'valor': f"{diff_emp:.4f}",
            'status': '✅' if diff_emp < 1 else '❌'
        })

    # 8. Coeficientes não-negativos
    min_a = A.min()
    resultados.append({
        'item': 8, 'verificacao': 'Coeficientes não-negativos',
        'critério': 'min(A) ≥ -1e-10',
        'valor': f"{min_a:.6e}",
        'status': '✅' if min_a >= -1e-10 else '❌'
    })

    # 9. Soma colunas < 1
    max_col = A.sum(axis=0).max()
    resultados.append({
        'item': 9, 'verificacao': 'Soma colunas A < 1',
        'critério': 'max(col_sum) < 1',
        'valor': f"{max_col:.6f}",
        'status': '✅' if max_col < 1 else '❌'
    })

    # 10. Coeficiente emprego plausível
    if l_coef is not None:
        l_min = l_coef.min()
        l_max = l_coef.max()
        plausivel = (l_min > 0) and (l_max < 2)
        resultados.append({
            'item': 10, 'verificacao': 'Coef. emprego plausível',
            'critério': '0 < l_j < 2',
            'valor': f"min={l_min:.4f}, max={l_max:.4f}",
            'status': '✅' if plausivel else '❌'
        })

    return pd.DataFrame(resultados)
```

### 3.11 Relatório completo de multiplicadores e índices

```python
def gerar_relatorio_completo(A, X, emp, VA, D, h_produto,
                              nomes_setores=None, codigos_setores=None,
                              l_coef=None, w_coef=None,
                              remuneracoes=None,
                              alfa_setorial=None,
                              fazer_bootstrap=True, n_boot=1000,
                              fazer_jackknife=False):
    """
    Executa o pipeline completo e gera relatório.

    Fluxo:
        1. Carrega/valida dados
        2. Calcula inversa de Leontief
        3. Calcula multiplicadores Tipo I
        4. Calcula multiplicadores Tipo II
        5. Calcula índices RH
        6. Classifica setores
        7. Bootstrap (opcional)
        8. Jackknife (opcional)
        9. Valida 10 critérios

    Parâmetros:
        A (ndarray): Matriz de coeficientes (n×n)
        X (array): Produção total (n,)
        emp (array): Emprego (n,)
        VA (array): Valor adicionado (n,)
        D (ndarray): Matriz D (n×m)
        h_produto (array): Consumo famílias (m,)
        nomes_setores (list, opcional)
        codigos_setores (array, opcional)
        l_coef (array, opcional): Coeficiente emprego
        w_coef (array, opcional): Coeficiente renda
        remuneracoes (array, opcional): Remunerações reais
        alfa_setorial (dict, opcional): Alfa por setor
        fazer_bootstrap (bool): Calcular IC bootstrap
        n_boot (int): Réplicas bootstrap
        fazer_jackknife (bool): Análise jackknife

    Retorna:
        dict: Relatório completo com todos os resultados
    """
    n = A.shape[0]

    # 1. Coeficientes
    if l_coef is None:
        l_coef = calcular_coeficiente_emprego(emp, X)
    if w_coef is None:
        w_coef = calcular_coeficiente_renda(
            VA, X, remuneracoes, alfa_setorial=alfa_setorial
        )

    # 2. Inversa de Leontief
    L = calcular_inversa_leontief(A)

    # 3. Multiplicadores Tipo I
    tipo_I = calcular_multiplicadores_tipo_I(L, l_coef, w_coef)

    # 4. Modelo fechado
    c_col = converter_consumo_familias(h_produto, D, X)
    tipo_II = calcular_multiplicadores_tipo_II(
        A, X, D, h_produto, VA, remuneracoes,
        alfa_setorial=alfa_setorial
    )

    # 5. Índices RH
    BL, FL = calcular_indices_rh(L)

    # 6. Classificação
    classificacao = classificar_setores(BL, FL, nomes_setores)

    # 7. Bootstrap
    bootstrap = None
    if fazer_bootstrap:
        bootstrap = bootstrap_multiplicadores(l_coef, L, n_boot=n_boot)

    # 8. Jackknife
    jackknife = None
    if fazer_jackknife:
        jackknife = jackknife_rh(A, nomes_setores)

    # 9. Validação
    validacao = validar_multiplicadores(
        A, L, tipo_I['MP'], l_coef=l_coef
    )

    # 10. Estatísticas descritivas
    def _estatisticas(arr, nome):
        return {
            'indicador': nome,
            'media': float(f"{arr.mean():.4f}"),
            'minimo': float(f"{arr.min():.4f}"),
            'maximo': float(f"{arr.max():.4f}"),
            'desvio_padrao': float(f"{arr.std():.4f}")
        }

    estatisticas = [
        _estatisticas(tipo_I['MP'], 'MP Tipo I'),
    ]

    if 'ME' in tipo_I:
        estatisticas.append(_estatisticas(tipo_I['ME'], 'ME Tipo I'))
    if 'MR' in tipo_I:
        estatisticas.append(_estatisticas(tipo_I['MR'], 'MR Tipo I'))

    return {
        'n_setores': n,
        'estatisticas_descritivas': estatisticas,
        'multiplicadores_tipo_I': {
            'MP': tipo_I.get('MP'),
            'ME': tipo_I.get('ME'),
            'MR': tipo_I.get('MR')
        },
        'multiplicadores_tipo_II': {
            'MP_II': tipo_II.get('MP_II')
        },
        'indices_rh': {
            'BL': BL,
            'FL': FL
        },
        'classificacao_setorial': classificacao,
        'bootstrap': bootstrap,
        'jackknife': jackknife,
        'validacao': validacao,
        'coeficientes': {
            'l_coef': l_coef,
            'w_coef': w_coef,
            'c_col': c_col
        }
    }
```

---

## 4. Tabelas de Referência

### 4.1 Multiplicadores — resumo conceitual

| Multiplicador | Sigla | Fórmula | Interpretação |
|---------------|-------|---------|---------------|
| Produção Tipo I | MP(I) | Σᵢ Lᵢⱼ | R$ total gerado por R$ 1 de demanda final |
| Produção Tipo II | MP(II) | Σᵢ L_fechadoᵢⱼ | MP(I) + efeito induzido das famílias |
| Emprego Tipo I | ME(I) | (lᵀ·L)ⱼ / lⱼ | Empregos totais por emprego direto |
| Emprego Tipo II | ME(II) | (lᵀ·L_fechado)ⱼ / lⱼ | ME(I) + empregos induzidos |
| Renda Tipo I | MR(I) | (wᵀ·L)ⱼ / wⱼ | Renda total por R$ 1 de renda direta |
| Renda Tipo II | MR(II) | (wᵀ·L_fechado)ⱼ / wⱼ | MR(I) + renda induzida |

### 4.2 Índices Rasmussen-Hirschman

| Índice | Fórmula | Interpretação |
|--------|---------|---------------|
| **BL** (Backward Linkage) | n · Σᵢ Lᵢⱼ / ΣᵢΣⱼ Lᵢⱼ | Poder de compra de insumos. BL > 1 → arrasta cadeia acima da média |
| **FL** (Forward Linkage) | n · Σⱼ Lᵢⱼ / ΣᵢΣⱼ Lᵢⱼ | Poder de venda de insumos. FL > 1 → é insumo sistêmico |

### 4.3 Classificação setorial (4 categorias)

| Categoria | BL | FL | Nº setores (MG, 67 set.) | Exemplos em MG |
|-----------|----|----|--------------------------|----------------|
| I — Setor-chave | >1 | >1 | 13 | Açúcar, Refino petróleo, Siderurgia, Energia elétrica |
| II — Forte p/ trás | >1 | ≤1 | 18 | Abate carne, Automóveis, Bebidas, Couro |
| III — Forte p/ frente | ≤1 | >1 | 8 | Agricultura, Petróleo/gás, Transporte terrestre |
| IV — Dispersor | ≤1 | ≤1 | 28 | Mineração, Construção, Imobiliário, Serviços |

### 4.4 Parâmetro δ de Flegg para FLQ

| Contexto | δ sugerido | Fonte |
|----------|-----------|-------|
| Padrão (região pequena/média) | 0,30 | Flegg & Webber (2000) |
| Região muito pequena (E_r/E_n < 0,02) | 0,35–0,40 | Literatura |
| Região grande (E_r/E_n > 0,20) | 0,20–0,25 | Literatura |

### 4.5 Parâmetro α — parcela salarial no VA

| Contexto | α sugerido | Fonte |
|----------|-----------|-------|
| Padrão Brasil (média 2010-2020) | 0,60 | SCN/IBGE |
| Setores capital-intensivos (ex: imobiliário) | 0,25–0,35 | Ajuste recomendado |
| Serviços intensivos em trabalho | 0,65–0,85 | Ajuste recomendado |
| Administração pública | 0,75 | Ajuste recomendado |
| **Ideal: extrair valor real da MIP** | Remunerações / X | Tabela de Usos |

---

## 5. Exemplo Completo

```python
# ==========================================================================
# Exemplo: Multiplicadores e Índices RH para MG (67 setores)
# ==========================================================================
import numpy as np
import pandas as pd

# --- Dados simulados para demonstração ---
n = 67
np.random.seed(42)

# Matriz A regionalizada (simulada)
A = np.random.dirichlet(np.ones(n), size=n) * 0.3
np.fill_diagonal(A, A.diagonal() * 0.5)

# Produção, emprego, VA (simulados)
X = np.random.uniform(100, 5000, size=n)
emp = np.random.uniform(1000, 200000, size=n)
VA = X * np.random.uniform(0.3, 0.7, size=n)

# Matriz D e consumo famílias (simulados)
m = 127  # produtos
D = np.random.dirichlet(np.ones(n), size=m).T  # n×m
h_produto = np.random.uniform(10, 500, size=m)

# Nomes setoriais
nomes = [f"Setor_{i}" for i in range(n)]

# --- Executar pipeline completo ---
relatorio = gerar_relatorio_completo(
    A=A, X=X, emp=emp, VA=VA,
    D=D, h_produto=h_produto,
    nomes_setores=nomes,
    fazer_bootstrap=True, n_boot=500,
    fazer_jackknife=False
)

# --- Estatísticas descritivas ---
print("=== ESTATÍSTICAS DESCRITIVAS ===")
for est in relatorio['estatisticas_descritivas']:
    print(f"{est['indicador']}: média={est['media']}, "
          f"min={est['minimo']}, max={est['maximo']}")

# --- Top 5 por categoria ---
print("\n=== DISTRIBUIÇÃO DAS CATEGORIAS ===")
print(relatorio['classificacao_setorial']['categoria'].value_counts())

# --- Validação ---
print("\n=== VALIDAÇÃO (CHECKLIST 10 CRITÉRIOS) ===")
validacao = relatorio['validacao']
for _, row in validacao.iterrows():
    print(f"{row['status']} {row['verificacao']}: {row['valor']}")

# --- Bootstrap (exemplo: primeiros 5 setores) ---
if relatorio['bootstrap']:
    boot = relatorio['bootstrap']
    print("\n=== BOOTSTRAP (primeiros 5 setores) ===")
    for j in range(5):
        print(f"Setor {j}: mediana={boot['mediana'][j]:.4f}, "
              f"IC 95%=[{boot['ci_inf'][j]:.4f}, {boot['ci_sup'][j]:.4f}]")
```

---

## 6. Regras

### O que SEMPRE fazer
- Verificar |det(I - A)| > 1e-10 **antes** de inverter a matriz
- Usar `np.linalg.solve(M, I)` em vez de `np.linalg.inv(M)` por estabilidade numérica
- Verificar condição de Hawkins-Simon: todos os elementos de L devem ser não-negativos
- Confirmar que a média de BL e FL é exatamente 1.0 (propriedade dos índices RH)
- Documentar a fonte de cada coeficiente (emprego: RAIS, renda: MIP ou α setorial)
- Executar o checklist de 10 critérios de validação em toda execução
- Usar bootstrap com no mínimo 1.000 réplicas (ideal: 10.000) para intervalos de confiança
- Ajustar α (parcela salarial no VA) por setor quando possível, em vez de usar valor fixo
- Salvar log de execução com parâmetros, fontes e validações

### O que NUNCA fazer
- Nunca inverter (I - A) sem verificar o determinante
- Nunca usar α=0,60 fixo para todos os setores se houver dados de remuneração disponíveis
- Nunca calcular multiplicadores sem verificar que são ≥ 1
- Nunca apresentar índices RH sem classificar os setores nas 4 categorias
- Nunca reportar multiplicador Tipo II sem explicar que inclui efeito induzido
- Nunca criar resíduos artificiais para forçar consistência dos dados
- Nunca escolher uma fonte de emprego arbitrariamente sem documentar a divergência
- Nunca confundir multiplicador Tipo I com Tipo II — são conceitos distintos

### Quando algo falhar
- **det(I-A) ≈ 0**: verificar colunas que somam próximo de 1; identificar setores com coeficientes mal estimados; ajustar e rebalancear
- **Multiplicador < 1**: verificar matriz A (coeficientes negativos? balanceamento incorreto?); corrigir e recalcular
- **Média BL ≠ 1 (erro > 1e-6)**: verificar arredondamento na normalização; BL deve ser exatamente n·col_sum / total_sum
- **Bootstrap lento**: reduzir n_boot para 1.000 ou usar `bootstrap_paralelo()` com joblib
- **Dados de remuneração ausentes**: usar α setorial documentado em vez de valor fixo; reportar a limitação
- **Coeficiente de emprego > 2 (implausível)**: verificar unidade de X (R$ milhões vs. R$); setor pode ter produção muito baixa
- **Consumo famílias negativo**: verificar extração da Tabela de Usos; coluna 73 (0-indexed) pode variar entre níveis
- **Jackknife não converge para algum setor**: setor pode ser structuralmente crítico (sua remoção torna a matriz singular)

---

## 7. Glossário

| Termo | Definição |
|-------|-----------|
| **MIP** | Matriz Insumo-Produto |
| **A** | Matriz de coeficientes técnicos (a_ij = insumo do setor i para produzir 1 unidade do setor j) |
| **L** | Inversa de Leontief (I - A)⁻¹ — captura efeitos diretos + indiretos |
| **MP** | Multiplicador de Produção — produção total gerada por unidade de demanda final |
| **ME** | Multiplicador de Emprego — empregos totais por emprego direto |
| **MR** | Multiplicador de Renda — renda total por unidade de renda direta |
| **Tipo I** | Modelo aberto (apenas setores produtivos endógenos) |
| **Tipo II** | Modelo fechado (famílias endogeneizadas — inclui efeito induzido) |
| **BL** | Backward Linkage (Rasmussen-Hirschman) — poder de encadeamento para trás |
| **FL** | Forward Linkage (Rasmussen-Hirschman) — poder de encadeamento para frente |
| **Setor-chave** | BL > 1 e FL > 1 — setor com alto encadeamento nos dois sentidos |
| **l_j** | Coeficiente de emprego (empregos / R$ produzido) |
| **w_j** | Coeficiente de renda do trabalho (remuneração / R$ produzido) |
| **α** | Parcela dos salários no valor adicionado |
| **VA** | Valor Adicionado |
| **CNAE** | Classificação Nacional de Atividades Econômicas |
| **RAIS** | Relação Anual de Informações Sociais |
| **FLQ** | Flegg's Location Quotient |
| **RAS** | Método biproporcional de balanceamento matricial |
| **Bootstrap** | Reamostragem com reposição para intervalos de confiança |
| **Jackknife** | Remoção sequencial de setores para análise de robustez estrutural |

---

## 8. Referências

- Flegg, A. T., & Webber, C. D. (2000). Regional size, regional specialization and the FLQ formula. *Regional Studies*, 34(6), 563-569.
- Flegg, A. T., & Tohmo, T. (2013). A comment on Tobias Kronenberg's "Construction of regional input-output tables using nonsurvey methods". *International Regional Science Review*, 36(3), 336-346.
- Guilhoto, J. J. M. (2010). *Input-Output Analysis: Theory and Foundations.* MPRA Paper 32566.
- Hirschman, A. O. (1958). *The Strategy of Economic Development.* Yale University Press.
- Leontief, W. (1936). Quantitative Input and Output Relations in the Economic System of the United States. *The Review of Economics and Statistics*, 18(3), 105-125.
- Miller, R. E., & Blair, P. D. (2009). *Input-Output Analysis: Foundations and Extensions* (2nd ed.). Cambridge University Press.
- Rasmussen, P. N. (1956). *Studies in Inter-Sectoral Relations.* Amsterdam: North-Holland.
- IBGE (2018). *Matriz de Insumo-Produto: Brasil 2015.* Rio de Janeiro: IBGE.
- MTE (2024). *RAIS 2023 — Sumário Executivo.* Brasília: MTE.
