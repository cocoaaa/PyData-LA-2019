conda create -n mlexp python=3.7
conda activate mlexp
conda install -c pyviz holoviz

#optional: to use jupyterlab instead of jupyter notebook
#conda install -c conda-forge jupyterlab
#jupyter labextension install @pyviz/jupyterlab_pyviz

jupyter notebook #or, jupyter-lab
