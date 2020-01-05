# Abstract
## Tutorial-PyData-LA-2019

This tutorial introduces how to make your data exploration and model building process more interactive and exploratory by using the combination of JupyterLab, HoloViews, and PyTorch. HoloViews is a set of Python libraries that offers simple yet powerful visualization and GUI building tools which, together with other data analysis libraries (eg. pandas, geopandas, numpy) and machine learning framework (eg. PyTorch, Tensorflow).  

I will start by introducing four core HoloViews libraries (Holoviews, GeoViews, Panel and Param) and guide you through how to directly interact with your visualization by eg. hovering over the graph to inspect values, querying RGB values of an image, or Lat/Lon values on your map.

Following the introduction, I will show how you can turn your PyTorch codes into a simple GUI that encaptulates the state of your model (or alternatively, the state of your training session). This GUI explicitly exposes your model parameters and training hyperparameters (eg. learning rate, optimizer settings, batch size) as interactively controllable parameters. Compared to conventional ways of specifying the hyperparameter settings with the help of 'argparse' library or config files, this GUI-based approach focuses on the experimental nature of modeling and integrates seamlessly with Jupyter notebooks. After training a neural network model using our own GUI in the notebook, I will demonstrate how to understand a trained model by visualizing the intermediate layers with HoloViews and test the model with test images directly sampled from HoloViews visualization.

To illustrate these steps, I will focus on the problem of classfying different types of roads on satellite images, defined as a multi-class semantic segmentation problem. Starting from the data exploration to the trained model understanding, you will learn different ways to explore the data and models by easily building simple GUIs in a Jupyter notebook.

In summary, by the end of the talk you will have learned:

- how to make your data exploration more intuitive and experimental using HoloViews libraries
- how to turn your model script into a simple GUI that allows interactive hyperparameter tuning and model exploration
- how to monitor the training process in realtime
- how to quickly build a GUI tool to inspect the trained models in the same Jupyter notebook

