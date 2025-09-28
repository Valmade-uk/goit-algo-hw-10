#!/usr/bin/env python3
"""
Завдання 1: Розмін суми монетами.

Містить дві функції:
- find_coins_greedy(amount, coins): жадібний алгоритм.
- find_min_coins(amount, coins): динамічне програмування (оптимальна кількість монет).

Також є прості тести та мікро-бенчмарк.
"""
from __future__ import annotations
from typing import Dict, List
import random
import time

DEFAULT_COINS = [50, 25, 10, 5, 2, 1]


def _to_dict(counts: Dict[int, int]) -> Dict[int, int]:
    """Повертає словник без номіналів з нульовою кількістю."""
    return {c: n for c, n in sorted(counts.items(), reverse=True) if n > 0}


def find_coins_greedy(amount: int, coins: List[int] = None) -> Dict[int, int]:
    """
    Жадібний розмін: на кожному кроці беремо найбільший можливий номінал.
    Повертає словник {номінал: кількість}.
    """
    if coins is None:
        coins = DEFAULT_COINS
    coins = sorted(coins, reverse=True)

    result: Dict[int, int] = {c: 0 for c in coins}
    remaining = amount
    for c in coins:
        if remaining <= 0:
            break
        k, remaining = divmod(remaining, c)
        if k:
            result[c] = k
    if remaining != 0:
        raise ValueError("Неможливо розміняти суму цими монетами")
    return _to_dict(result)


def find_min_coins(amount: int, coins: List[int] = None) -> Dict[int, int]:
    """
    Динамічне програмування: мінімальна кількість монет для точної суми.
    Складність: O(amount * len(coins)) за часом та O(amount) за пам'яттю.
    Повертає словник {номінал: кількість}.
    """
    if coins is None:
        coins = DEFAULT_COINS
    coins = sorted(coins)  # зростання зручніше для DP
    INF = 10**9

    # dp[s] = мінімальна кількість монет для суми s
    dp = [0] + [INF] * amount
    # prev[s] = попередня сума після вибору монети coin_used[s]
    prev = [-1] * (amount + 1)
    coin_used = [-1] * (amount + 1)

    for c in coins:
        for s in range(c, amount + 1):
            if dp[s - c] + 1 < dp[s]:
                dp[s] = dp[s - c] + 1
                prev[s] = s - c
                coin_used[s] = c

    if dp[amount] >= INF:
        raise ValueError("Неможливо розміняти суму цими монетами")
    # Відновлення відповіді
    res: Dict[int, int] = {c: 0 for c in coins}
    s = amount
    while s > 0:
        c = coin_used[s]
        if c == -1:
            raise RuntimeError("Відновлення рішення не вдалося")
        res[c] += 1
        s = prev[s]
    return _to_dict(res)


def _self_test():
    print("=== Самоперевірка ===")
    for amount in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 113, 999]:
        g = find_coins_greedy(amount)
        d = find_min_coins(amount)
        print(f"{amount:4d}: greedy={g}  dp={d}")
        # Для канонічних систем (як у задачі) greedy == dp
        assert sum(k*v for k, v in g.items()) == amount
        assert sum(k*v for k, v in d.items()) == amount
        assert sum(g.values()) == sum(d.values())

    print("ОК!")


def _micro_benchmark(trials: int = 5000, max_amount: int = 5000):
    print("\n=== Мікро-бенчмарк ===")
    random.seed(42)
    amounts = [random.randint(1, max_amount) for _ in range(trials)]

    t0 = time.perf_counter()
    for a in amounts:
        find_coins_greedy(a)
    t1 = time.perf_counter()

    for a in amounts:
        find_min_coins(a)
    t2 = time.perf_counter()

    print(f"Greedy: {t1 - t0:.4f} c; DP: {t2 - t1:.4f} c; Співвідношення DP/Greedy = {(t2 - t1)/(t1 - t0):.1f}x")


if __name__ == "__main__":
    _self_test()
    _micro_benchmark()
