from src.general import *
import pandas as pd
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pd.options.plotting.backend = 'plotly'
pio.templates.default = 'plotly_dark'

import colorcet as cc
px.defaults.color_continuous_scale = cc.bjy

px.defaults.color_discrete_sequence = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
sky_blue = "#56B4E9"
orange = "#E69F00"
green = "#009E73"

px.defaults.width = 1600
px.defaults.height = 600

heatmap = {'kind' : 'imshow', 'aspect' : 'auto'}


def lineup_plots(plots):
    
    fig = make_subplots(cols=len(plots), rows=1) 

    for i, plot in enumerate(plots):
        for trace in range(len(plot["data"])):
            fig.append_trace(plot["data"][trace], col=i+1, row=1)

    fig.show()


def plot_patches(patches):
    
    df = patches.reset_index(drop=True)

    mask = np.zeros_like(df.corr(), dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    df.T.plot(**heatmap).show()
    lineup_plots([df.mean().sort_values().plot.barh().update_yaxes(categoryorder='total descending'),
                  df.sum(axis=1).value_counts(ascending=False, normalize=True).plot.bar()
    ])
    df.corr().mask(mask).plot(**heatmap).show()
    df.cov().mask(mask).plot(**heatmap).show()
    
    return df


def clean_names(patches):
    patches.columns = patches.columns.str.capitalize()
    patches.columns = patches.columns.str.replace('_', ' ')

    return patches


def use_proxy_species(phylo: pd.DataFrame, species_dictionary: dict):

    for missing_species, proxy_species in species_dictionary.items():
        phylo.loc[missing_species] = phylo.loc[proxy_species]
        phylo.loc[:, missing_species] = phylo.loc[:, proxy_species]

    return phylo


def sort_species(patches):
    
    species = patches.columns
    species_by_abundance = patches[species].mean().sort_values(ascending=False).index
    return patches[species_by_abundance]


def remove_boring(patches, cutoff = 1.5):
    
    return patches[patches.sum(axis='columns') >= cutoff]#.reset_index(drop=True)


def keep_species(patches, n_keep: int = 16):
    
    return patches.pipe(sort_species).iloc[:, :n_keep]


def crop_clean(patches, n_keep: int = 16, cutoff: float = 1.5):

    return patches.pipe(sort_species).pipe(keep_species, n_keep).pipe(remove_boring, cutoff).pipe(sort_species).reset_index(drop=True)


def export_patches(patches: pd.DataFrame, paths: dict, file_name: str='', index : str=False):
        
    paths['outputs'].mkdir(parents=True, exist_ok=True)
    path = paths['outputs'] / file_name
    patches.to_csv(str(path), index=index)
    return path


def get_patches(config: dict,
                data : str = 'testing_data'
                ):
    
    csvs = [p for p in (get_git_root() / 'outputs' / config[data]).glob('*.csv')]
    csv = csvs[-1]
    print(csv)
    df = pd.read_csv(csv)
    # display(df)
    patches = df
    if config['species_order'] != 'phylogeny':
        patches = (patches
                        .pipe(sort_species)
                        .pipe(keep_species, config['n_species'])
                        .pipe(remove_boring, config['min_species'])
                        .pipe(sort_species)
                  )

        if config['species_order'] == 'reverse_abundance':
            patches = patches[reversed(list(patches.columns))]

        if config['species_order'] == 'random':
            patches = patches.sample(frac=1, axis=1)
            
        else:
            pass
    
    return patches


def get_species(location : 'str'):
    
    paths = set_paths(location)
    csvs = [p for p in paths['outputs'].glob('*.csv')]
    csv = csvs[0]
    if len(csvs) > 1:
        print(f'WARNING: List of CSVs found in  directory:\n{paths["outputs"]}\nUsing {csv}\nPlease make sure this is correct')
    return read_data(csv).pipe(sort_species).pipe(clean_names).columns


def subset_phylo(phylo : pd.DataFrame,
                 species_origin : list,
                 species_target : list,
                 from_species : int = 0,
                 to_species : int = 16):
    phylo_subset = phylo.loc[:, species_origin[from_species:to_species]].reindex(species_target).iloc[from_species:to_species, :]
    # phylo_subset.plot(kind='imshow', height = 800).show()
    return phylo_subset


def match_and_remove(phylo_subset : pd.DataFrame,
                     sp1 : str,
                     sp2 : str,
                     dictionary : dict):
    dictionary[sp2] = sp1
    phylo_subset = phylo_subset.drop(index=sp1).drop(columns=sp2)
    return phylo_subset, dictionary


def build_dictionary(config : dict,
                     phylo : pd.DataFrame,
                     species_origin : list,
                     species_target : list,
                     chunk_size : int = 16,
                     multi_chunks : bool = False):

    dictionary = {}
    
    if multi_chunks != True:
        
        for i in range(chunk_size):
            dictionary[species_origin[i]] = species_target[i] 
        
        phylo_subset = subset_phylo(phylo, species_origin, species_target, from_species = chunk_size, to_species = None)
        
        # # first loop to find exact matches
        for sp in phylo_subset.columns:
            try:
                phylo_subset, dictionary = match_and_remove(phylo_subset, sp, sp, dictionary)
                # print(f'{sp} : {sp}')
            except:
                continue

        # #second loop to find closest matches
        for sp in phylo_subset.columns:
            closest_species = phylo_subset.index[phylo_subset[sp].argmin()]
            phylo_subset, dictionary = match_and_remove(phylo_subset, closest_species, sp, dictionary)
            # print(f'{closest_species} : {sp}')

        
    else:
        n_chunks = config['n_species'] // chunk_size + 1
        remaining_species = config['n_species'] % chunk_size


        for n_chunk in range(0, n_chunks):

            # display(phylo)
            phylo_subset = subset_phylo(phylo, species_origin, species_target, from_species = n_chunk * chunk_size, to_species = (n_chunk + 1) * chunk_size)

            if n_chunk == n_chunks - 1:
                chunk_size = remaining_species
                if remaining_species == 0:
                    continue
            else:
                chunk_size = chunk_size

            # print(f'\nMatching chunk number {n_chunk} with {chunk_size} species\n')

            # # first loop to find exact matches
            for sp in phylo_subset.columns:
                try:
                    phylo_subset, dictionary = match_and_remove(phylo_subset, sp, sp, dictionary)
                    # print(f'{sp} : {sp}')
                except:
                    continue

            # #second loop to find closest matches
            for sp in phylo_subset.columns:
                closest_species = phylo_subset.index[phylo_subset[sp].argmin()]
                phylo_subset, dictionary = match_and_remove(phylo_subset, closest_species, sp, dictionary)
                # print(f'{closest_species} : {sp}')


    return dictionary