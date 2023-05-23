from pathlib import Path
import subprocess
import pandas as pd

def configure_defaults() -> dict:
    config = {}
    config['n_species'] = 16
    config['min_species'] = 2
    config['species_order'] = 'abundance'
    config['training_data'] = 'petrer_limestone'
    config['testing_data'] = 'petrer_limestone'
    
    config['training_start'] = 'train_from_scratch'

    config['val_split'] = 0.50
    config['reconstruction_loss'] = 'bce'
    config['depth'] = 32
    config['n_latent'] = 16
    config['beta'] = 1
    config['monitor'] = 'val_loss'
    config['mode'] = 'min'

    config['learning_rate'] = 1e-3
    config['batch_size'] = 64
    config['epochs'] = 200
    config['stride'] = 2

    config['n_bootstrap'] = 30

    config['n_tile'] = 8
    config['n_colors'] = 1
    config['fully_connected'] = False
    config['testing'] = False
    
    config['chunk_size'] = 1
    
    return config


def get_git_root() -> Path:

    return Path(subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8'))


def create_dir(path : Path):

    path.mkdir(parents=True, exist_ok=True)


def set_paths(location: str) -> dict:

    root = get_git_root()
    paths = {'inputs' : root / 'inputs' / location,
            'outputs' : root / 'outputs' / location,
            'figures' : root / 'figures' / location,
            'models' : root / 'models' / location}
    create_dir(paths['outputs'])
    # [create_dir(p) for p in paths.values()]
    return paths 


def get_model_path(config: dict) -> Path:

    model_path = get_git_root() / 'models' / config['reconstruction_loss'] / config['monitor'] / config['training_data'] / config['training_start'] / ('n_species='+str(config['n_species'])) / ('depth=' + str(config['depth'])) / ('n_latent=' + str(config['n_latent'])) / ('val_split={:.2f}'.format(config['val_split']))
    model_path.mkdir(exist_ok=True, parents=True)
    
    return model_path


def show(config: dict):
    
    out = pd.DataFrame([config]).T
    out.columns = ['value']
    
    return out