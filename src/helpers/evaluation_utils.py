import torch
import os
import pandas as pd
from src.commands.graph_learning.main import Config, Trainer, HeteroData, Graph, Dict, Any
from typing import Tuple
from definitions import EXPERIMENTS_DIR

def load_data(
    experiment_id : str,
    config_id : str
) -> Tuple[HeteroData, Graph]:

    config = Config(experiment=experiment_id)
    exper_config = config.training_config.configs[config_id]

    trainer = Trainer(
        experiment=experiment_id, 
        config=config.training_config,
        configs=None
    )

    data, graph = trainer.load_data(config=exper_config, return_raw=True)
    split = trainer.create_or_load_split(data=data)

    data['Article'].split = split

    return data, graph

def load_history(experiment_id : str) -> pd.DataFrame:

    histories_dir = os.path.join(EXPERIMENTS_DIR, experiment_id, 'histories')
    histories = []

    for filename in os.listdir(histories_dir):

        if "_final" in filename:
            continue

        history_path = os.path.join(histories_dir, filename)
        history = pd.read_csv(history_path)
        history['trial'] = int(os.path.splitext(filename)[0][-1])
        histories.append(history)

    return pd.concat(histories)

def aggregate_history(history : pd.DataFrame) -> pd.DataFrame:

    history = history.groupby(by=['epoch', 'phase', 'config', 'trial']) \
        .mean().reset_index()

    history_mean = history \
        .drop(columns = 'trial') \
        .groupby(['epoch','phase','config']).mean() \
        .add_suffix('_mean')

    history_std = history \
        .drop(columns = 'trial') \
        .groupby(['epoch','phase','config']).std() \
        .add_suffix('_std')

    return history_mean.join(other = history_std).reset_index()

def load_results(experiment_id : str) -> Dict[Tuple[str, int], Any]:

    results_dir = os.path.join(EXPERIMENTS_DIR, experiment_id, 'results')
    results = {}

    for filename in os.listdir(results_dir):

        if "_final" in filename:
            continue

        config_id = filename.split('_')[1]
        trial = int(os.path.splitext(filename)[0][-1])
        results_path = os.path.join(results_dir, filename)
        results[(config_id, trial)] = torch.load(results_path, weights_only=False)

    return results