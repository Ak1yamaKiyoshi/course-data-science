import pandas as pd

df = pd.read_csv('bert_ner_moountain_data.csv', delimiter=";", error_bad_lines=False)

df.describe()

