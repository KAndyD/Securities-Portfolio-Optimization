"""Проверка выполняется по исходным данным, без использования состояния солвера."""
import numpy as np
from model import matrices

TOL = 1e-7


def check(data, allocation):
    x = np.asarray(allocation, dtype=float)
    income, risk, limits = matrices(data)
    if x.shape != income.shape or not np.all(np.isfinite(x)):
        raise ValueError('Некорректный вектор распределения')
    observed = risk @ x
    violations = {'budget': abs(float(np.sum(x)) - 1.0),
                  'lower_bounds': float(max(0, -np.min(x))),
                  'upper_bounds': float(max(0, np.max(x - data['max_weight']))),
                  'risk_limits': float(max(0, np.max(observed - limits)))}
    return {'ok': all(v <= TOL for v in violations.values()),
            'violations': violations, 'risk_values': observed.tolist(),
            'expected_yield_pct': float(income @ x),
            'expected_income_rub': float(income @ x * data['capital_rub'] / 100)}
