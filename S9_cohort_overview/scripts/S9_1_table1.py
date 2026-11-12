"""Table 1. Cohort characteristics of the discovery cohort.

Input
  metadata/metadata_discovery.csv   Age, Sex, Diet, Drug, Disease_pr of the 3,659 participants
Output (S9_cohort_overview/results/)
  Table1_cohort_characteristics.csv   characteristic, value

Process
  Age: mean ± SD, median (IQR) and range. Sex, disease status, diet and medication use: counts and percentages of
  the 3,659 participants.

Usage
  python S9_1_table1.py
"""
import os

import pandas as pd

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
META = os.path.join(PKG, 'metadata', 'metadata_discovery.csv')
RES = os.path.join(PKG, 'S9_cohort_overview', 'results')

DIET = [('3_Balanced', 'Balanced'), ('2_Meat_Oriented', 'Mostly Carnivorous'), ('1_Meat_Exclusive', 'Carnivorous biased'),
        ('4_Veget_Oriented', 'Mostly Vegetarian'), ('5_Veget_Exclusive', 'Vegetarian biased')]
DRUG = [('1_No_Drug', 'None'), ('7_Others', 'Others'), ('3_Painkillers', 'Painkillers'), ('5_Digestive_Aids', 'Digestive aids'),
        ('2_Antibiotics', 'Antibiotics')]


def main():
    m = pd.read_csv(META)
    n = len(m)
    cnt = lambda col, level: f'{(m[col] == level).sum():,} ({(m[col] == level).mean() * 100:.1f}%)'
    age = m['Age']
    rows = [('Total sample size', f'{n}'),
            ('Age, mean ± SD', f'{age.mean():.1f} ± {age.std():.1f} years'),
            ('Age, median (IQR)', f'{age.median():.0f} ({age.quantile(0.25):.0f}–{age.quantile(0.75):.0f}) years'),
            ('Age range', f'{age.min()}–{age.max()} years'),
            ('Sex, Female / Male', f"{cnt('Sex', '2_Female')} / {cnt('Sex', '1_Male')}"),
            ('Disease status (none / any)', f"{cnt('Disease_pr', '1_No')} / {cnt('Disease_pr', '2_Yes')}")]
    rows += [(f'Diet — {label}', cnt('Diet', code)) for code, label in DIET]
    rows += [(f'Drug use — {label}', cnt('Drug', code)) for code, label in DRUG]
    rows += [('Drug use — Laxatives / Diarrheals', f"{cnt('Drug', '4_Laxatives')} / {cnt('Drug', '6_Diarrheals')}")]
    t = pd.DataFrame(rows, columns=['Characteristic', 'Value'])
    os.makedirs(RES, exist_ok=True)
    t.to_csv(os.path.join(RES, 'Table1_cohort_characteristics.csv'), index=False)
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
