"""Решение портфельной ЛП-модели и обязательные сценарные эксперименты."""
import copy
import csv
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from model import load_data, matrices, solve, status
from verify import check

ROOT = Path(__file__).parent
OUT = ROOT / 'results'


def csv_write(name, columns, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    original = load_data()
    scenarios = {}
    logs = []
    for label in ('Базовый', 'Дефицитный', 'Стрессовый'):
        data = copy.deepcopy(original)
        if label in ('Дефицитный', 'Стрессовый'):
            data['risk_limits'][0] *= 0.8
        if label == 'Стрессовый':
            data['risk_limits'][2] *= 0.8
        result = solve(data)
        row = {'scenario': label, 'status': status(result), 'message': result.message,
               'risk_limits': data['risk_limits'], 'iterations': int(result.nit or 0)}
        logs.append(f'{label}: {result.message}; simplex_iterations={row["iterations"]}')
        if result.success:
            verification = check(data, result.x)
            if not verification['ok']:
                raise RuntimeError(f'Независимая проверка не пройдена: {label}')
            row.update({'weights': result.x.tolist(), 'yield_pct': verification['expected_yield_pct'],
                        'income_rub': verification['expected_income_rub'], 'check': verification,
                        'shadow_prices_per_risk_unit': (-result.ineqlin.marginals).tolist(),
                        'slack': result.ineqlin.residual.tolist()})
        scenarios[label] = row
    base = scenarios['Базовый']
    if base['status'] != 'optimal':
        raise RuntimeError(f'Базовая задача не решена: {base["message"]}')
    csv_write('portfolio.csv', ['Актив', 'Доля', 'Вложение (руб)', 'Доходность (%/год)'],
              [[item['name'], round(w, 10), round(w * original['capital_rub'], 2), item['yield_pct']]
               for item, w in zip(original['assets'], base['weights'])])
    csv_write('scenarios.csv', ['Сценарий', 'Статус', 'Доходность (%/год)', 'Доход (руб/год)', 'Пределы рисков'],
              [[key, value['status'], value.get('yield_pct'), value.get('income_rub'),
                json.dumps(value['risk_limits'], ensure_ascii=False)] for key, value in scenarios.items()])
    (OUT / 'verification.json').write_text(json.dumps(base['check'], ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'scenarios.json').write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'solver.log').write_text(f'Python: {platform.python_version()}\nSciPy: {scipy.__version__}\n'
                                    'Метод: HiGHS dual simplex (highs-ds); presolve=true\n'
                                    + '\n'.join(logs) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh([a['name'] for a in original['assets']], base['weights'])
    ax.set_xlabel('Доля капитала')
    ax.set_title('Оптимальная структура портфеля')
    fig.tight_layout()
    fig.savefig(OUT / 'portfolio.png', dpi=170)
    plt.close(fig)
    lines = ['# Отчёт: портфель ценных бумаг', '', 'Группа Б24-507, номер 14, вариант 5, seed 10014.', '',
             '## Задача и модель', '', 'Распределить 1 000 000 рублей между шестью активами, максимизируя ожидаемый годовой доход.',
             'Переменная xᵢ — доля капитала в активе i. Цель: max Σ rᵢxᵢ (%/год).',
             'Ограничения: Σxᵢ = 1; 0 ≤ xᵢ ≤ 0,45; для каждого k=1..3: Σaₖᵢxᵢ ≤ Rₖ.',
             'Три риск-показателя — безразмерные экспертные баллы, а не волатильность или вероятность убытка.',
             'Линейная модель не учитывает корреляции доходностей, комиссии, налоги и реальную случайность рынка.', '',
             '## Результат и проверка', '',
             f'Базовый статус: **{base["status"]}**, доходность {base["yield_pct"]:.6f}%/год, '
             f'ожидаемый доход {base["income_rub"]:.2f} руб/год.',
             f'Проверка NumPy: **{base["check"]["ok"]}**, '
             f'максимальная ошибка: {max(base["check"]["violations"].values()):.3g}.', '',
             '| Актив | Доля капитала |', '|---|---:|']
    lines += [f'| {a["name"]} | {x:.6f} |' for a, x in zip(original['assets'], base['weights'])]
    lines += ['', '## Сценарии', '', '| Сценарий | Статус | Ожидаемая доходность, % |', '|---|---|---:|']
    lines += [f'| {k} | {v["status"]} | {v.get("yield_pct", float("nan")):.6f} |' for k, v in scenarios.items()]
    lines += ['', 'Дефицит: допустимый рыночный риск уменьшен на 20%. Стресс: дополнительно на 20% уменьшен предел риска ликвидности.',
              'Если ужесточение приводит к infeasible, нельзя приводить план из другой постановки как допустимый.', '',
              '## Теневые цены', '',
              'Для активного ограничения малая прибавка ΔRₖ к пределу даёт прогноз Δдоходности ≈ πₖΔRₖ.',
              'Этот прогноз локальный: уменьшение на 20% может сменить базис, поэтому сравнение с фактическим результатом обязательно.', '',
              '| Ограничение | Запас | Теневая цена, п.п. на единицу риска |', '|---|---:|---:|']
    lines += [f'| {name} | {slack:.6f} | {price:.6f} |' for name, slack, price in
              zip(original['risk_names'], base['slack'], base['shadow_prices_per_risk_unit'])]
    for name in ('Дефицитный', 'Стрессовый'):
        sc = scenarios[name]
        if sc['status'] == 'optimal':
            delta_limits = np.asarray(sc['risk_limits']) - np.asarray(original['risk_limits'])
            forecast = float(np.dot(base['shadow_prices_per_risk_unit'], delta_limits))
            actual = sc['yield_pct'] - base['yield_pct']
            lines.append(f'{name}: линейный прогноз по всем изменённым пределам Δr={forecast:.6f} п.п.; '
                         f'фактическое изменение Δr={actual:.6f} п.п.')
    lines += ['', 'Нулевые доли возникают, если увеличение вложений в актив невыгодно при текущих рисках, '
              'ограничениях концентрации и альтернативных доходностях.',
              'Модель годится для учебной оценки компромисса доходности и риска; практический портфель потребует '
              'оценки вероятностного риска, ликвидности и актуальных рыночных данных.', '',
              'Решатель: SciPy linprog / HiGHS dual simplex, presolve включён. '
              'Для оптимального решения проверяется допустимость, а не только код статуса.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('ЛР1 завершена:', base['status'], f'{base["yield_pct"]:.4f}%/год')


if __name__ == '__main__':
    main()
