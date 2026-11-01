"""Непрерывная линейная модель распределения капитала."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog


def load_data(path=None):
    path = Path(path) if path else Path(__file__).parent / 'data' / 'portfolio.json'
    return json.loads(path.read_text(encoding='utf-8'))


def matrices(data):
    assets = data['assets']
    coefficients = np.array([a['yield_pct'] for a in assets], dtype=float)
    risk = np.array([a['risk'] for a in assets], dtype=float).T
    limits = np.asarray(data['risk_limits'], dtype=float)
    return coefficients, risk, limits


def solve(data):
    income, risk, limits = matrices(data)
    return linprog(-income, A_ub=risk, b_ub=limits,
                   A_eq=np.ones((1, len(income))), b_eq=[1.0],
                   bounds=[(0, data['max_weight'])] * len(income),
                   method='highs-ds', options={'presolve': True, 'primal_feasibility_tolerance': 1e-8,
                                               'dual_feasibility_tolerance': 1e-8})


def status(result):
    return {0: 'optimal', 1: 'time_or_iteration_limit', 2: 'infeasible',
            3: 'unbounded', 4: 'numerical_failure'}.get(result.status, 'unknown')
