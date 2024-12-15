import pandas
import sklearn.metrics

import cltrier_lib

DATA_FILE: str = "/home/s2shsinh/data/processed/DefaktS_Twitter.binary.csv"
N_SAMPLES: int = 500

dataset: pandas.DataFrame = (
    pandas.read_csv(DATA_FILE, index_col=[0])
    .replace(dict(binary_label={0.0: "neutral_post", 1.0: "possible_fake_news"}))
    .sample(n=N_SAMPLES)
)
print(dataset.head())