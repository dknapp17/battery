
from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def main():

    PROJECT_ROOT = next(
        p for p in Path(__file__).resolve().parents
        if (p / ".git").exists()
    )

    print("Hello from battery!")
    dataset = "razanihababdellatif/battery-remaining-useful-life-rul-dataset"
    obs_path = "battery_observations.csv"

    metadata_path = "battery_metadata.csv"


    battery_obs = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset,
        obs_path,
    )

    battery_metadata = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset,
        metadata_path,
    )

    batteries_all = battery_obs.merge(battery_metadata, on="battery_id")
    batteries_all.head()

    batteries_all['timestamp'] = pd.to_datetime(batteries_all['timestamp'])

    train_data = batteries_all.loc[batteries_all["dataset_split"]=="Train"]

    validation_data = batteries_all.loc[batteries_all["dataset_split"]=="Validation"]

    test_data = batteries_all.loc[batteries_all["dataset_split"]=="Test"]

    print('writing data to /data/')
    train_data.to_csv(f'{PROJECT_ROOT}/data/train_batteries.csv', index=False)
    validation_data.to_csv(f'{PROJECT_ROOT}/data/validation_batteries.csv', index=False)
    test_data.to_csv(f'{PROJECT_ROOT}/data/test_batteries.csv', index=False)

    print('write complete')
    
if __name__ == "__main__":
    main()
