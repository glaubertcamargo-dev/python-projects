"""
Estabilidade Local de Sistemas Dinâmicos Fuzzy
==============================================
Implementação numérica do método de Mizukoshi, Barros & Bassanezi
(Biomatemática XIII, 2003).

Ideia central (Proposição 1 do artigo):
    [φ̂_t(X0)]^α = φ_t([X0]^α)
ou seja, cada α-nível da solução fuzzy é a imagem do α-nível inicial
pelo fluxo clássico da EDO x'(t) = f(x(t)).

Em dimensão 1 o fluxo é monótono na condição inicial (trajetórias não se
cruzam), então basta integrar os dois extremos de cada α-nível.

Requisitos:  pip install numpy scipy matplotlib
Uso:         python estabilidade_fuzzy.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Números fuzzy triangulares e seus α-níveis
# ---------------------------------------------------------------------------
class NumeroFuzzyTriangular:
    """Número fuzzy triangular (a, b, c), com a <= b <= c."""

    def __init__(self, a, b, c):
        if not (a <= b <= c):
            raise ValueError("É preciso ter a <= b <= c.")
        self.a, self.b, self.c = a, b, c

    def alfa_nivel(self, alfa):
        """Retorna o intervalo [x1, x2] do α-nível."""
        return (self.a + alfa * (self.b - self.a),
                self.c - alfa * (self.c - self.b))

    def __repr__(self):
        return f"Fuzzy({self.a:.3f}, {self.b:.3f}, {self.c:.3f})"


def niveis_crisp(x):
    """α-níveis do número crisp χ{x} (todos iguais a [x, x])."""
    return lambda alfa: (x, x)


# ---------------------------------------------------------------------------
# 2. Fluxo clássico e fluxo fuzzy
# ---------------------------------------------------------------------------
def fluxo_classico(f, x0, tempos):
    """Resolve x' = f(x), x(0) = x0, e devolve x(t) nos instantes dados."""
    sol = solve_ivp(lambda t, x: f(x), (tempos[0], tempos[-1]), [x0],
                    t_eval=tempos, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"Falha na integração a partir de x0={x0}: {sol.message}")
    return sol.y[0]


def fluxo_fuzzy(f, X0, tempos, alfas):
    """
    Aplica a Proposição 1: integra os extremos de cada α-nível.
    Retorna um array de forma (len(alfas), 2, len(tempos)):
        [i, 0, :] -> extremo inferior do α-nível i ao longo do tempo
        [i, 1, :] -> extremo superior do α-nível i ao longo do tempo
    """
    resultado = np.empty((len(alfas), 2, len(tempos)))
    for i, alfa in enumerate(alfas):
        x1, x2 = X0.alfa_nivel(alfa)
        traj1 = fluxo_classico(f, x1, tempos)
        traj2 = traj1 if x1 == x2 else fluxo_classico(f, x2, tempos)
        # garante a ordem inferior/superior (fluxo pode ser decrescente)
        resultado[i, 0, :] = np.minimum(traj1, traj2)
        resultado[i, 1, :] = np.maximum(traj1, traj2)
    return resultado


# ---------------------------------------------------------------------------
# 3. Métrica D (supremo das distâncias de Hausdorff entre α-níveis)
# ---------------------------------------------------------------------------
def hausdorff_intervalos(u, v):
    """Distância de Hausdorff entre intervalos [u1,u2] e [v1,v2]."""
    return max(abs(u[0] - v[0]), abs(u[1] - v[1]))


def distancia_D(niveis_u, niveis_v, alfas):
    """D(u, v) = sup_α h([u]^α, [v]^α), aproximado na grade de α."""
    return max(hausdorff_intervalos(niveis_u(a), niveis_v(a)) for a in alfas)


def curva_D(trajetoria_fuzzy, equilibrio, alfas):
    """D(φ̂_t(X0), X̄0) para cada instante de tempo."""
    n_t = trajetoria_fuzzy.shape[2]
    D = np.empty(n_t)
    for k in range(n_t):
        D[k] = max(
            hausdorff_intervalos(trajetoria_fuzzy[i, :, k], equilibrio(a))
            for i, a in enumerate(alfas)
        )
    return D


# ---------------------------------------------------------------------------
# 4. Testes de equilíbrio e de estabilidade
# ---------------------------------------------------------------------------
def eh_equilibrio_fuzzy(f, X_eq, tempos, alfas, tol=1e-6):
    """Def. 3: φ̂_t(X̄0) = X̄0 para todo t (verificado numericamente)."""
    traj = fluxo_fuzzy(f, X_eq, tempos, alfas)
    D = curva_D(traj, X_eq.alfa_nivel, alfas)
    return D.max() < tol, D.max()


def perturbacoes_aleatorias(x_eq, delta, n, rng):
    """Gera n números fuzzy triangulares com D(X0, χ{x_eq}) < delta."""
    perturb = []
    for _ in range(n):
        a, b, c = np.sort(rng.uniform(-delta, delta, 3) * 0.999)
        perturb.append(NumeroFuzzyTriangular(x_eq + a, x_eq + b, x_eq + c))
    return perturb


def testar_estabilidade(f, x_eq, tempos, alfas, epsilon, delta,
                        n_amostras=30, tol_assintotica=1e-3, semente=0):
    """
    Def. 4 (verificação numérica):
      - estável: D(φ̂_t(X0), χ{x_eq}) < ε para todo t amostrado,
                 sempre que D(X0, χ{x_eq}) < δ;
      - assintoticamente estável: além disso, D -> 0 no fim do horizonte.
    Atenção: isto é evidência numérica, não demonstração.
    """
    rng = np.random.default_rng(semente)
    equilibrio = niveis_crisp(x_eq)
    curvas, condicoes = [], perturbacoes_aleatorias(x_eq, delta, n_amostras, rng)

    estavel, assintotico = True, True
    for X0 in condicoes:
        try:
            traj = fluxo_fuzzy(f, X0, tempos, alfas)
        except RuntimeError:
            estavel = assintotico = False
            continue
        D = curva_D(traj, equilibrio, alfas)
        curvas.append(D)
        if not np.all(np.isfinite(D)) or D.max() >= epsilon:
            estavel = False
        if not np.isfinite(D[-1]) or D[-1] > tol_assintotica:
            assintotico = False

    return {
        "estavel": estavel,
        "assintoticamente_estavel": estavel and assintotico,
        "curvas_D": curvas,
        "condicoes_iniciais": condicoes,
    }


# ---------------------------------------------------------------------------
# 5. Visualização
# ---------------------------------------------------------------------------
def plotar(titulo, tempos, traj, alfas, resultado, epsilon, arquivo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    cmap = plt.get_cmap("viridis")
    for i, a in enumerate(alfas):
        ax1.fill_between(tempos, traj[i, 0], traj[i, 1],
                         color=cmap(a), alpha=0.25, linewidth=0)
    ax1.plot(tempos, traj[-1, 0], "k-", lw=1.5, label="núcleo (α = 1)")
    ax1.plot(tempos, traj[0, 0], "k--", lw=0.8, label="suporte (α = 0)")
    ax1.plot(tempos, traj[0, 1], "k--", lw=0.8)
    ax1.set_title(f"{titulo}\nEvolução dos α-níveis da solução fuzzy")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x(t)")
    ax1.legend()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax1, label="α")

    for D in resultado["curvas_D"]:
        ax2.plot(tempos, D, color="tab:blue", alpha=0.35, lw=1)
    ax2.axhline(epsilon, color="tab:red", ls="--", label=f"ε = {epsilon}")
    ax2.set_yscale("log")
    ax2.set_title("D(φ̂ₜ(X₀), equilíbrio) para várias condições iniciais")
    ax2.set_xlabel("t")
    ax2.set_ylabel("D (escala log)")
    ax2.legend()

    status = ("assintoticamente estável" if resultado["assintoticamente_estavel"]
              else "estável" if resultado["estavel"] else "instável")
    fig.suptitle(f"Diagnóstico numérico: {status}", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=130)
    print(f"  gráfico salvo em {arquivo}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Experimentos
# ---------------------------------------------------------------------------
def experimento(nome, f, x_eq, X0_exemplo, t_final, epsilon, delta, arquivo):
    print(f"\n=== {nome} ===")
    alfas = np.linspace(0, 1, 21)
    tempos = np.linspace(0, t_final, 400)

    ok, erro = eh_equilibrio_fuzzy(f, NumeroFuzzyTriangular(x_eq, x_eq, x_eq),
                                   tempos, alfas)
    print(f"  χ{{{x_eq}}} é equilíbrio fuzzy? {ok}  (desvio máximo {erro:.2e})")

    res = testar_estabilidade(f, x_eq, tempos, alfas, epsilon, delta)
    print(f"  estável (ε={epsilon}, δ={delta})?        {res['estavel']}")
    print(f"  assintoticamente estável?            {res['assintoticamente_estavel']}")

    traj = fluxo_fuzzy(f, X0_exemplo, tempos, alfas)
    print(f"  condição inicial de exemplo: {X0_exemplo}")
    print(f"  α-nível 0 em t={t_final}: [{traj[0,0,-1]:.4g}, {traj[0,1,-1]:.4g}]")
    plotar(nome, tempos, traj, alfas, res, epsilon, arquivo)


if __name__ == "__main__":
    # Exemplo 2 do artigo: Malthus com decaimento, x' = -a x (a > 0)
    a = 0.8
    experimento("Malthus  x' = -a·x  (a = 0.8)",
                f=lambda x: -a * x, x_eq=0.0,
                X0_exemplo=NumeroFuzzyTriangular(-0.4, 0.1, 0.5),
                t_final=10, epsilon=0.5, delta=0.5,
                arquivo="malthus_estavel.png")

    # Contraexemplo: crescimento, x' = +a x  -> a origem é instável
    experimento("Malthus  x' = +a·x  (a = 0.8)",
                f=lambda x: a * x, x_eq=0.0,
                X0_exemplo=NumeroFuzzyTriangular(-0.05, 0.0, 0.05),
                t_final=6, epsilon=0.5, delta=0.05,
                arquivo="malthus_instavel.png")

    # Modelo sem a mesma estrutura linear: logístico, x' = r x (1 - x/K)
    # equilíbrio x = K é assintoticamente estável
    r, K = 1.0, 10.0
    experimento("Logístico  x' = r·x(1 - x/K)  (r = 1, K = 10)",
                f=lambda x: r * x * (1 - x / K), x_eq=K,
                X0_exemplo=NumeroFuzzyTriangular(7.0, 8.0, 9.5),
                t_final=12, epsilon=3.0, delta=2.0,
                arquivo="logistico.png")