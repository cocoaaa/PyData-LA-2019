conda create -y -n expml python=3.7
conda activate expml

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y

conda install -y -c pyviz holoviews bokeh
conda install -y -c conda-forge pandas
conda install -y -c conda-forge tqdm
conda install -y -c conda-forge joblib
conda install -y -c conda-forge xarray
conda install -y -c conda-forge scikit-image

conda install -y -c conda-forge widgetsnbextension ipywidgets

pip install torchsummary

# #optional: to use jupyterlab instead of jupyter notebook, we additionally ned these
# conda install -y -c conda-forge jupyterlab
# jupyter labextension install @pyviz/jupyterlab_pyviz
# ## ipywidgets
# conda install -c conda-forge widgetsnbextension ipywidgets
# jupyter labextension install @jupyter-widgets/jupyterlab-manager


jupyter notebook #or, jupyter-lab

