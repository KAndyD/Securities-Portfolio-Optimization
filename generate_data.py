"""Фиксированный учебный набор данных для варианта 5."""
import json
from pathlib import Path
import numpy as np

SEED = 10014
PATH = Path(__file__).parent / 'data' / 'portfolio.json'


def generate():
    rng = np.random.default_rng(SEED)
    names = ['Облигации', 'Дивидендные акции', 'Акции роста', 'Корпоративные облигации', 'Фонд денежного рынка', 'Технологический фонд']
    returns = [5.8, 7.5, 9.8, 11.6, 6.8, 12.2]
    risk = [[0.8, 1.6, 1.2], [1.1, 1.1, 1.4], [1.4, 0.8, 0.9],
            [1.8, 1.5, 1.6], [1.0, 0.7, 1.1], [2.2, 2.0, 2.1]]
    assets = []
    for name, income, scores in zip(names, returns, risk):
        assets.append({'name': name, 'yield_pct': round(income + rng.uniform(-0.12, 0.12), 3),
                       'risk': scores})
    return {'variant': 5, 'group': 'Б24-507', 'student_number': 14, 'seed': SEED,
            'capital_rub': 1000000, 'assets': assets,
            'risk_names': ['Рыночный риск', 'Кредитный риск', 'Риск ликвидности'],
            'risk_limits': [1.46, 1.33, 1.36], 'max_weight': 0.45}


if __name__ == '__main__':
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(generate(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(PATH)
