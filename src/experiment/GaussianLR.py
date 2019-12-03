from abc import abstractmethod

import numpy as np
import torch
from tqdm.autonotebook import tqdm
from scipy.stats import multivariate_normal

import holoviews as hv
from holoviews import opts

from .datagen import make_linear_data 
# vs. 
# from experiment.datagen import make_linear_data

from datasets import TabularDataset
from utils import to_device


#todo:
class DataGenerator:
    @abstractmethod
    def split_fn(self):
        pass

#todo: make it compatible with DataGenerator class
class GaussianLR_Data(DataGenerator):
    """
    Gaussian source distribution (ie. data samples (x's) are generated from a multivariate gaussian 
    and linking function (ie. f s.t y = f(x)) is linear. more accurate to say 'affine'
    
    x ~ Gauss(mu, cov) #multivariate gaussian
    y = f(x) = Wx + b + noise
    
    where noise ~ Gauss(noise_mu, noise_std) #1dim gaussian, each sample's noise is independent of other samples' noises
    
    
    We create one such dataset, which is a dict with k,v pairs of:
        key='X', value=np.ndarray of shape (n_samples, n_features)
    
    Returns the dataset
    """
    def __init__(self, mu, cov, noise_mu, noise_var, n_samples, true_w,true_b, 
                 allow_singular=True, seed=None, label='' ):
        self.label = label
        self.seed = seed or np.random.randint(2**10)
        self.mu, self.cov = mu, cov
        self.noise_mu, self.noise_var = noise_mu, noise_var
        self.allow_singular = allow_singular
        
        self.n_samples = n_samples
        self.n_features = len(true_w)
        assert self.n_features == len(self.mu) and self.n_features == len(self.cov), (
            'Dimension of weights and x_generating distribution must match: '
            f'{self.n_features} vs. {len(self.mu)}, {self.n_features}'
        )
        
        self.true_w, self.true_b = true_w, true_b
        
        self.data = self.get_data()
        
    @property
    def is_noise_free(self):
        return np.isclose(self.noise_mu, 0) and np.isclose(self.noise_var, 0)
    
    @property
    def is_noisy(self):
        return not self.is_noise_free
    
    @property
    def noise_descr(self):
        return (f'G({self.noise_mu}, {self.noise_var})')
    
    @property
    def descr(self):
        return (f'Gaussian-mean:{self.mu}-cov:{self.cov}'
                f'_LR-w:{self.true_w}-b:{self.true_b}'
                f'_Noise-mean:{self.noise_mu}-var:{self.noise_var}'
               f'_Seed:{self.seed}')
    @property
    def shortname(self):
        return 'GaussianLR_Noisy' if self.is_noisy else 'GaussianLR_Noisefree'
    
    @property
    def x_gen(self):
        return multivariate_normal(mean=self.mu, cov=self.cov, 
                                   allow_singular=self.allow_singular, seed=self.seed).rvs
    
    @property
    def noise_gen(self):
        return multivariate_normal(mean= self.noise_mu, cov=self.noise_var, 
                                   allow_singular=self.allow_singular, seed=self.seed).rvs #1dim
    @property
    def split_fn(self):
        """Function defining how to split an output into x and y (if needed)"""
        return lambda batch: (batch['X'], batch['y'])
    
    def get_data(self):
        """
        Returns y data as a result of x->y mapping by matrix-multiplying X with ture_w + true_b
        with optional additive gaussian noises

        Noise-free dataset is generated if noise_mu = 0 and noise_var = 0
        """
        data = make_linear_data(self.x_gen, self.n_samples, self.true_w, self.true_b, self.noise_gen)
        return data
    
    def hvplot(self, scatter_opt=None):
        if self.n_features != 2:
            raise NotImplementedError
        hv.extension('plotly')
#         x, y = self.data['X'], self.data['y']
        x, y = self.split_fn(self.data)
        scatter_opt = scatter_opt or opts.Scatter3D(size=5,  
                                                    width=800, 
                                                    height=800, 
                                                    title=self.descr,
                                                   show_legend=True if self.label else False)
        

        return hv.Scatter3D((x[:,0], x[:,1], y.squeeze()), label=self.label).opts(scatter_opt)

    def get_pytorch_dataset(self):
        
        return TabularDataset(self.data)

    
    
class GaussianLR_DataBunch():
    """
    Gaussian source distribution (ie. data samples (x's) are generated from a multivariate gaussian 
    and linking function (ie. f s.t y = f(x)) is linear. more accurate to say 'affine'
    
    x ~ Gauss(mu, cov) #multivariate gaussian
    y = f(x) = Wx + b + noise
    
    where noise ~ Gauss(noise_mu, noise_std) #1dim gaussian, each sample's noise is independent of other samples' noises
    
    
    We create two such dataset, one for training data and the other for the test data.
    Each dataset is a dict with k,v pairs of:
        key='X', value=np.ndarray of shape (n_samples, n_features)
    
    Returns the two datasets in a dictionary of keys 'train' and 'test
    """
    def __init__(self,
        train_mu, train_cov, train_noise_var, 
        test_mu, test_cov, test_noise_var,
        n_samples,true_w,true_b, 
        train_noise_mu = 0,
         test_noise_mu = 0,
         allow_singular=True,
        seed=None
    ):
        self.seed = seed
        self.train_mu, self.train_cov = train_mu, train_cov
        self.train_noise_mu, self.train_noise_var = train_noise_mu, train_noise_var

        self.test_mu, self.test_cov = test_mu, test_cov
        self.test_noise_mu, self.test_noise_var = test_noise_mu, test_noise_var    
        self.allow_singular = allow_singular
        
        self.n_samples = n_samples
        self.true_w, self.true_b = true_w, true_b

    @property
    def train_gen(self):
        self.train_gen = GaussianLR_Data(self.train_mu, self.train_cov, self.train_noise_mu, 
                                         self.train_noise_var, self.n_samples,self.true_w, self.true_b, 
                                         self.allow_singular, self.seed)
        return self.train_gen

    @property
    def test_gen(self):
        self.test_gen = gen = GaussianLR_Data(self.test_mu, self.test_cov, self.test_noise_mu, 
                                              self.test_noise_var, self.n_samples, self.true_w, 
                                              self.true_b, self.allow_singular, self.seed)
        return self.test_gen
        
    @property
    def train_data(self):
        return self.train_gen.get_data()
    
    @property
    def test_data(self):
        return self.test_gen.get_data()
                                      
    def get_databunch(self):
        train_xy = self.train_data
        test_xy = self.test_data
        
        return {'train': train_xy, 'test': test_xy}
        

def get_demo_2d_gaussian_lr_data_gens(train_noise_var=0.1, test_noise_var=0, n_samples=100,
                   true_w=[1,2], true_b=10, seed=None):
    train_mu = [0,0]
    train_cov = [[1,0],[0,1]]
    train_noise_mu = 0
    train_noise_var = train_noise_var

    test_mu = [10,10]
    test_cov = [[1,0], [0,1]]
    test_noise_mu = 0
    test_noise_var = test_noise_var

    n_samples = 100
    true_w = true_w
    true_b = true_b

    train_data_gen = GaussianLR_Data(train_mu, train_cov, train_noise_mu, train_noise_var, 
                                     n_samples, true_w, true_b, seed=seed, label='train')
    test_data_gen = GaussianLR_Data(test_mu, test_cov, test_noise_mu, test_noise_var, 
                                    n_samples, true_w, true_b, seed=seed, label='test')
    return {'train': train_data_gen,
            'test': test_data_gen}

## Train
def train_one_epoch(model, train_dl, optimizer, loss_fn, device, metric_fns=None):
    """
    Returns epoch's average loss (ie. loss per sample) and other metrics
    """
    to_device(model, device)
    model.train()
    result = {}
    
    # Average loss from the training this epochs
    epoch_loss = 0.
    sample_count = 0
    for sample in train_dl:
        x, y = sample['X'].to(device), sample['y'].to(device)
        y_pred = model(x)
        loss_mean = loss_fn(y_pred, y)

        # Backprop
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        epoch_loss += loss_mean.item() * len(x)
        sample_count += len(x)

    # Collect epoch stats
    epoch_loss /= sample_count
    result['epoch_loss'] = epoch_loss
    ## add other metrics
    # for mname, metric_fn in metric_fns.items():
    # ...
    
    return result


def train(n_epochs, model, optimizer, loss_fn, train_dl, tbw, losses=None, verbose=False):
    """
    Assumes model and optmizer are properly linked, 
    ie. optimizer's parameters == model.parameters()
    - loss_fn should use the loss averaged over the mini-batch
    
    - If losses is not None, append each epoch's loss to the input `losses`. 
        Otherwise, create a new empty list and append to it.
        
    - tbw (TBWrapper of tensorboard.writer.SummaryWriter)
    """
    assert_mean_reduction(loss_fn)
    if losses is None: losses = []
        
    # Log weight and bias before new trianing runs
    global_ep = len(losses) 
    log_first_grad = (global_ep > 0)
 
    if verbose:
        print('Start training at ep: ', global_ep)
        print('Logging to ep idx: ', global_ep)
    tbw.log_params_hist(model, global_ep, log_grad=log_first_grad)
    global_ep += 1

    for ep in range(n_epochs):
        
        # Run train for one epoch and collect epoch stats
        epoch_result = train_one_epoch(model, optimizer, loss_fn, train_dl, device)
        epoch_loss = epoch_result['epoch_loss']
        losses.append(epoch_loss)
        
        # Log to tensorboard
#         print('Logging to ep idx: ', global_ep)
        tb.add_scalar('train/loss', epoch_loss, global_ep)
        
        ## Log weight
        tbw.log_params_hist(model, global_ep, log_grad=True)
        global_ep += 1
        
        if verbose:
            print('number of samples in this epoch: ', sample_count)
        ## Log train and val (and optionally test) metrics
        ## todo
            
    return losses

def evaluate(model, val_dl, loss_fn, device, metric_fns=None):
    """
    Evaluate the model on val_dl using the input loss_fn and specified parameters
    on `device`
    
    Good references:
    - https://is.gd/EP8LKv
    - use yaml file for config parsing: [pytorch-semseg](https://is.gd/Gbcq4H)
    
    Args:
    - val_dl (DataLoader): 
        - validation dataloader
        
    - params (dict)
    
    Returns:
    - result(dict) : {'loss': val_loss, 'acc': val_acc}
    """
    
    # Get parameters

    # Put model on the right device
    model = model.to(device)
    
    # Eval
    model.eval()
    
    total_loss = 0.
    n_samples = 0
    with torch.no_grad():
        for sample in tqdm(val_dl, desc='eval-iter'): #tqdm(range(max_epoch), desc='Epoch')
            x, y = sample['X'], sample['y']
            x,y = x.to(device), y.to(device)
            pred_y = model(x)

            # Eval loss
            loss_mean = loss_fn(pred_y, y)
            total_loss += loss_mean * len(x)
            n_samples += len(x)

            # Eval metrics
    
    result = {'eval_loss': total_loss/n_samples}
    # add  other metrics to result

    
    return result
          