from src.general import *
from src.eda import *
import keras
from keras.models import Model
from keras.layers import Dense,Input,Flatten,Reshape,Conv2DTranspose,Conv2D,Lambda
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import copy
from datetime import datetime
from IPython.display import clear_output
import tensorflow as tf
import matplotlib.pyplot as plt

n_vars_default = 16
n_tile_default = 8
n_colors_default = 1
stride_default = 2
default_min_diversity = 1.5
midtone_default = 0.5
threshold_default = 0.5
default_depth = 32
default_n_latent = 8
default_learning_rate = 1e-3

def prep_data(df: pd.DataFrame,
              n_tile : int=n_tile_default,
              n_colors: int=n_colors_default,
              fully_connected : bool = False
              ) -> np.array:

    X=np.array(df)

    if not fully_connected:
        X=np.tile(X,(n_tile,1,1))
        X=X.transpose(1,2,0)
        X=X[:,np.newaxis].transpose(0,2,3,1)
        if n_colors==3:
            X=X*[1,1,1]

    return X


def split_val(patches : pd.DataFrame,
              val_split : float = 0):
    
    if val_split == 0:
        train_df = patches
        val_df = patches
    else:
        train_df = patches.sample(frac=1-val_split)
        val_df = patches.iloc[patches.index.difference(train_df.index)].reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

#    display(train_df)
    train_ds = prep_data(train_df)
#    display(val_df)
    val_ds = prep_data(val_df)
    
    return train_ds, val_ds, train_df, val_df


def build_vae(config : dict):
    
    stride = config['stride']
    n_vars = config['n_species']
    learning_rate = config['learning_rate']
    n_tile = config['n_tile']
    n_colors = config['n_colors']
    depth = config['depth']
    n_latent = config['n_latent']
    fully_connected = config['fully_connected']
    testing = config['testing']
    beta = config['beta']
    
    
    def sampler(layers):
    
        z_mean, z_log_var = layers
        std_norm = K.random_normal(shape=(K.shape(z_mean)[0], n_latent), mean=0, stddev=1)

        return z_mean + K.exp(0.5 * z_log_var) * std_norm

    if not fully_connected:
        inp = Input(shape = (n_vars, n_tile, n_colors))
        x = inp
        x = Conv2D(depth,(2,2),strides = stride,activation = "relu",padding = "same")(x)
        x = Conv2D(2 * depth,(2,2),strides = stride,activation = "relu",padding = "same")(x)
        x = Conv2D(4 * depth,(2,2),strides = stride,activation = "relu",padding = "same")(x)
        shape = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(4 * depth, activation = "relu")(x)
        x = Dense(2 * depth, activation = "relu")(x)
        
    else:
        inp = Input(shape = (n_vars,))
        x = inp
        for _ in range(10):
            x = Dense(depth, activation = "relu")(x)

    z_mean = Dense(n_latent,activation = None)(x)
    z_log_var = Dense(n_latent,activation = None)(x)
    latent_vector = Lambda(sampler)([z_mean, z_log_var])
    
    encoder = Model(inp,latent_vector,name = "VAE_Encoder")
    
    decoder_inp = Input(shape = (n_latent,))
    x = decoder_inp
    
    if not fully_connected:
        x = Dense(2 * depth, activation = "relu")(x)
        x = Dense(4 * depth, activation = "relu")(x)
        x = Dense(shape[1]*shape[2]*shape[3],activation = "relu")(x)
        x = Reshape((shape[1],shape[2],shape[3]))(x)
        x = (Conv2DTranspose(4 * depth,(2,2),strides = stride,activation = "relu",padding = "same"))(x)
        x = (Conv2DTranspose(2 * depth,(2,2),strides = stride,activation = "relu",padding = "same"))(x)
        x = (Conv2DTranspose(depth,(2,2),strides = stride,activation = "relu",padding = "same"))(x)
        outputs = Conv2DTranspose(n_colors, (2,2), activation = 'sigmoid', padding = 'same', name = 'decoder_output')(x)
        
    else:
        for _ in range(10):
            x = Dense(depth,activation = "relu")(x)
            
            
        outputs = Dense(n_vars, activation='sigmoid')(x)
    
    decoder = Model(decoder_inp,outputs,name = "VAE_Decoder")
    
    vae = Model(inp,decoder(encoder(inp)),name = "Variational_Auto_Encoder")
    
    def vae_loss(input_img,
                output,
                beta : float=1):

        if config['reconstruction_loss'] == 'mse':
            reconstruction_loss = K.sum(K.square(output-input_img))
        elif config['reconstruction_loss'] == 'bce':
            reconstruction_loss = K.sum(tf.keras.losses.binary_crossentropy(input_img, output))
        
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        # return the average loss over all images in batch
        total_loss = K.mean(reconstruction_loss + beta * kl_loss)    

        return total_loss

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    vae.compile(optimizer = opt, loss = vae_loss, metrics = ["accuracy"])

    return vae, encoder, decoder


def compute_patch_errors(vae,
                   patches,
                   midtone = midtone_default,
                   threshold = threshold_default
                   ):
    
    Z_data = prep_data(patches)
    reals = Z_data[:,:,0,0]
    rec = vae.predict(Z_data)[:,:,:,0].mean(axis=2)

    binary_rec = rec#copy.deepcopy(rec)
    binary_rec = (np.sign(binary_rec-threshold)+1)*midtone
    errors = binary_rec-reals
    abs_errors = abs(errors)
    df_abs_errors = pd.DataFrame(abs_errors).T
    
    print('\nProportion of patches with a given number of mispredicted species:')
    
    return df_abs_errors.sum().value_counts(normalize=True)
    

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.i += 1
                
        clear_output(wait=True)
        
        print(f"Epoch {self.i}")

        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")  
        plt.plot(self.x, [np.array(self.val_losses).min()]*(self.i), color='black', linestyle='dashed', label="min_val_loss")
        plt.legend()
        plt.show()

        plt.plot(self.x, self.accuracy, label="accuracy")
        plt.plot(self.x, self.val_accuracy, label="val_accuracy")         
        plt.plot(self.x, [np.array(self.val_accuracy).max()]*(self.i), color='black', linestyle='dashed', label="max_val_accuracy")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
       

def binarize_vae_output(vae_out: np.array,
                        midtone : float=midtone_default,
                        threshold : float=threshold_default
                        ):
    
    rec = vae_out[:,:,:,0].mean(axis=2)
    binary_rec = copy.deepcopy(rec)

    binary_rec = (np.sign(binary_rec - threshold) + 1) * midtone
    
    return binary_rec


def quick_validate(vae,
                val_df: pd.DataFrame
                ):
    
    data = prep_data(val_df)
    # print(data.shape)
    pixel_accuracy = vae.evaluate(data, data, verbose=0)[1]                        
    reals = data[:,:,0,0]
    binary_rec = binarize_vae_output(vae.predict(data))
    errors = binary_rec - reals
    abs_errors = abs(errors)
    df_abs_errors = pd.DataFrame(abs_errors).reset_index(drop=True)
    results = df_abs_errors.sum(axis=1).gt(0).mean()
    pixel_err_rate = 1 - pixel_accuracy
    patch_err_rate = results
    
    return pixel_err_rate, patch_err_rate


def compute_errors(dfs,
                   model,
                   model_name : str = 'unknown_model_name'):
    
    patch_errors = pd.DataFrame()
    pixel_errors = pd.DataFrame()
    
    for data_name, patches in dfs.items():
        print(f'data: {data_name}')
        # print(species_order)
        temp, _, _ = validate_vae(vae, patches)
        temp.insert(0, 'data_name', data_name)
        temp.insert(0, 'model_name', model_name)
        
        patch_errors = pd.concat([patch_errors, temp], ignore_index=True)
        
        temp = compute_reco_errors(vae, patches)
        temp.insert(0, 'data_name', data_name)
        temp.insert(0, 'model_name', model_name)
        pixel_errors = pd.concat([pixel_errors, temp], ignore_index=True)
        
    return pixel_errors, patch_errors


def train_vae(config : dict,
              vae,
              train_ds : np.array,
              val_ds : np.array = None):    
    
    batch_size = config['batch_size']
    validation_split = config['val_split']
    epochs =config['epochs']
    monitor = config['monitor']
    mode = config['mode']
    
    checkpoint = ModelCheckpoint('checkpoint.ckpt', monitor=monitor, verbose=1, save_weights_only=False, save_best_only=True, mode=mode)
    callbacks_list = [checkpoint, plot_losses]
    vae.fit(train_ds, train_ds, batch_size = batch_size, epochs = epochs, validation_split = validation_split, shuffle = True, callbacks = callbacks_list, validation_data = (val_ds, val_ds))
 
    return vae


def train_once(vae,
              config : dict):    
           
    patches = get_patches(config, data = 'training_data')
    train_ds, val_ds, train_df, val_df = split_val(patches, val_split = config['val_split'])
    
    print(f'{train_ds.shape[0]} training samples\n{val_ds.shape[0]} validation samples')
    vae = train_vae(config, vae, train_ds, val_ds)

    return vae, train_ds, val_ds, train_df, val_df


def save_checkpoint(vae,
                    config : dict,
                    train_df : pd.DataFrame,
                    val_df : pd.DataFrame):

    vae.load_weights('checkpoint.ckpt')

    quick_validate(vae, val_df)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = get_model_path(config) / timestamp
    checkpoint_dir.mkdir(exist_ok=False, parents=True)
    
    print(f'Saving checkpoint and datasets in:\n{checkpoint_dir}')
    vae.save_weights( checkpoint_dir / 'vae.ckpt')
    train_df.to_csv(checkpoint_dir / 'train_df.csv')
    val_df.to_csv(checkpoint_dir / 'val_df.csv')
    
    return vae


def train_save(vae,
               config : dict
               ):
    
    vae, train_ds, val_ds, train_df, val_df = train_once(vae, config)
    save_checkpoint(vae, config, train_df, val_df)
    
    return vae


def train_save_multi(config : dict
                     ):

    for i in range(config['n_bootstrap']):
                
        vae, _, _ = build_vae(config)
        vae = train_save(vae, config)

        
def train_then_tune(config : dict,
               initial_data : str = 'petrer_limestone',
               tuning_data : str = 'union_patches'
              ):
    
    
    for i in tqdm(range(config['n_bootstrap'])):

        config['training_start'] = 'train_from_scratch'
        config['training_data'] = initial_data
        config['testing_data'] = initial_data
        config['epochs'] = 200

        print('Initial training')
        
        vae, _, _ = build_vae(config)
        vae = train_save(vae, config)
        
        config['training_start'] = 'tune_petrer_limestone'
        config['training_data'] = tuning_data
        config['testing_data'] = tuning_data
        config['epochs'] = 50

        print('Tuning')
        
        vae = train_save(vae, config)


def scan_train(config : dict,
               var : str = 'val_split',
               var_values : list = np.arange(.5, 1, .01)
               ):

    for var_value in tqdm(var_values):
        config[var] = var_value
        print(f'\nSetting {var} = {config[var]}')
        
        train_save_multi(config)


def load_checkpoint(config: dict,
                  checkpoint_path : Path):
    
    vae, encoder, decoder = build_vae(config)
    vae.load_weights(checkpoint_path)
    
    train_df = pd.read_csv(checkpoint_path.parents[0] / 'train_df.csv', index_col = 0)
    val_df = pd.read_csv(checkpoint_path.parents[0] / 'val_df.csv', index_col = 0)
    
    train_ds = prep_data(train_df)
    val_ds = prep_data(val_df)
        
    return vae, train_ds, val_ds, train_df, val_df


def evaluate_checkpoints(config : dict,
                        detailed_validation : bool =False):

    results = pd.DataFrame()

    checkpoint_paths = [p for p in get_model_path(config).rglob('*.ckpt')][:]
    n_checkpoints = len(checkpoint_paths)
    print(f'{n_checkpoints} checkpoints found, using {config["n_bootstrap"]} first checkpoints found')


    for checkpoint_path in tqdm(checkpoint_paths[:config['n_bootstrap']]):
        
        vae, train_ds, val_ds, train_df, val_df = load_checkpoint(config, checkpoint_path)
        
        if config['testing_data'] != config['training_data'] : #we're not testing on the original dataset
            patches = get_patches(config, data = 'testing_data')                
            train_ds, val_ds, train_df, val_df = split_val(patches, val_split = 0)

        if detailed_validation:
            temp_res, pix_err, patch_err = full_validate(vae, val_df)
           
        else:
            pix_err, patch_err = quick_validate(vae, val_df)
            temp_res = pd.DataFrame([{'dummy': 'dummy'}])

        for key, value in config.items():
            temp_res[key] = value
            
        temp_res['train_size'] = train_ds.shape[0]
        temp_res['val_size'] = val_ds.shape[0]

        temp_res['checkpoint_path'] = str(checkpoint_path)

        temp_res['pix_err'] = pix_err
        temp_res['patch_err'] = patch_err
        
        results = pd.concat([results, temp_res], ignore_index = True)
            
    return results


def compare_models(config : dict,
                   var : str,
                   var_values : list,
                   detailed_validation : bool = False
                  ):

    results = pd.DataFrame()

    for var_value in var_values:
        config[var] = var_value

        results = pd.concat([results, evaluate_checkpoints(config, detailed_validation)])
            
    return results


def compare_on_datasets(config,
                      testing_datas : list,
                      detailed_validation : bool =False
                      ):
    
    results = pd.DataFrame()
    
    for testing_data in testing_datas:
        
        config['testing_data'] = testing_data
        results = pd.concat([results, compare_models(config, 'testing_data', [testing_data], detailed_validation)])
        
    return results


def compare_species_order(config,
                         detailed_validation : bool =False):
    
    results = pd.DataFrame()
    
    config['training_data'] = 'petrer_limestone'
    config['testing_data'] = 'union_patches'

    for species_order in ['abundance', 'phylogeny', 'reverse_abundance']:

        config['species_order'] = species_order

        results = pd.concat([results, evaluate_checkpoints(config, detailed_validation)], ignore_index=True)

    return results


def compare_training(config,
                    detailed_validation : bool =False):
    
    results = pd.DataFrame()
    
    for testing_data in ['petrer_limestone', 'union_patches', 'mexico_all']:
        config['testing_data'] = testing_data
        config['training_data'] = testing_data

        temp = evaluate_checkpoints(config, detailed_validation)
        
        results = pd.concat([results, temp])
        
    return results


def compare_transfer(config,
                     testing_datas = ['union_patches', 'mexico_all'],
                     detailed_validation : bool =False):
    
    results = pd.DataFrame()

    for testing_data in testing_datas:
        
        config['training_start'] = 'train_from_scratch'
        config['model_type'] = 'Training from scratch'
        config['testing_data'] = testing_data
        config['training_data'] = testing_data
        results = pd.concat([results, evaluate_checkpoints(config, detailed_validation)])
        
        
        config['training_start'] = 'train_from_scratch'
        config['model_type'] = 'Direct transfer from Petrer'
        config['testing_data'] = testing_data
        config['training_data'] = 'petrer_limestone'
        results = pd.concat([results, evaluate_checkpoints(config, detailed_validation)])
            
        config['training_start'] = 'tune_petrer_limestone'
        config['model_type'] = 'Fine-tuning Petrer model'
        config['testing_data'] = testing_data
        config['training_data'] = testing_data
        results = pd.concat([results, evaluate_checkpoints(config, detailed_validation)])
                        
    return results