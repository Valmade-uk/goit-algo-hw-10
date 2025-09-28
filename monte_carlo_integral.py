#!/usr/bin/env python3
"""
Завдання 2: Обчислення визначеного інтеграла методом Монте-Карло
(усереднення значень функції).

Інтегруємо f(x)=x^2 на [0, 2] за замовчуванням.
Можна змінити функцію/межі/кількість точок через CLI.
"""
from __future__ import annotations
import argparse
import math
import random
from typing import Callable, Tuple

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


def f(x: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(x) ** 2


def mc_integral_avg(func: Callable, a: float, b: float, n: int, rng: random.Random | None = None) -> float:
    """
    Метод Монте-Карло (усереднення значень):
    ∫_a^b f(x) dx ≈ (b-a) * (1/n) * Σ f(x_i), x_i ~ U(a,b)
    """
    if rng is None:
        rng = random
    xs = np.array([rng.uniform(a, b) for _ in range(n)], dtype=float)
    return (b - a) * float(np.mean(func(xs)))


def analytic_integral_fx2(a: float, b: float) -> float:
    """Аналітичний інтеграл для f(x)=x^2: (b^3 - a^3)/3"""
    return (b**3 - a**3) / 3.0


def run(a: float, b: float, n: int, seed: int | None):
    rng = random.Random(seed)
    est = mc_integral_avg(f, a, b, n, rng)
    exact_quad, err_quad = spi.quad(lambda x: float(f(x)), a, b)
    analytic = analytic_integral_fx2(a, b)

    print(f"Монте-Карло (n={n}): {est:.8f}")
    print(f"quad: {exact_quad:.8f} (оцінка похибки {err_quad:.2e})")
    print(f"Аналітично: {analytic:.8f}")
    print(f"Абс. похибка MC vs quad: {abs(est - exact_quad):.6e}")
    print(f"Абс. похибка MC vs аналітичного: {abs(est - analytic):.6e}")

    # Побудова графіка (без явної вказівки кольорів, щоб зберегти стиль за замовч.)
    xs = np.linspace(a - 0.5, b + 0.5, 400)
    ys = f(xs)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, linewidth=2)
    ix = np.linspace(a, b, 200)
    iy = f(ix)
    ax.fill_between(ix, iy, alpha=0.3)
    ax.axvline(x=a, linestyle='--')
    ax.axvline(x=b, linestyle='--')
    ax.set_xlim([xs[0], xs[-1]])
    ax.set_ylim([0, max(ys) + 0.1])
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Інтегрування f(x)=x^2 на [{a}, {b}]; MC n={n}')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('mc_plot.png', dpi=150)
    print("Збережено графік: mc_plot.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--a', type=float, default=0.0, help='Нижня межа інтегрування')
    p.add_argument('--b', type=float, default=2.0, help='Верхня межа інтегрування')
    p.add_argument('--n', type=int, default=100000, help='Кількість випадкових точок')
    p.add_argument('--seed', type=int, default=42, help='Seed для відтворюваності')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.a, args.b, args.n, args.seed)
