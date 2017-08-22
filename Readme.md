# Kaggle Amazon deforestation challenge

This repository contains the source code of the [Kaggle Amazon deforestation challenge](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

# How to use and view the jupyter notebook
The jupyter notebook is stripped to a `.py` file for version control in the `notebook/` folder. To recreate the original `.ipynb` file and use it with jupyter execute `tools/extract_py_notebook.sh` (after cloning the repo for example).

When you'll work on the notebook, git will ignore the changes you made to it as specified in the `.gitignore` file. You need to strip it to a `.py` file first before pushing it to the repo.
To do so and let jupyter automatically create the `.py` file each time you save your notebook you need to add a script to its configuration simply by executing the following command:
```
tools/add_jupyter_vc.sh
```
This command should be only executed once on a system.
If you ever run into issues with this new configuration you can restore your original jupyter configuration with:
```
tools/restore_jupyter_config.sh
```
/!\ Be careful to not run `tools/add_jupyter_vc.sh` twice in a row as it will erase your original configuration file forever. 

# Dependencies

 - [Kaggle Data downloader](https://github.com/EKami/kaggle-data-downloader)
 - [Tensorflow 1.1](https://github.com/tensorflow/tensorflow/releases/tag/v1.1.0)
 - Scipy
 - Seaborn
 
 
 [Link to the associated gist preview of the notebook](https://gist.github.com/EKami/33ec0172590ab9f2e3a6b757c9f9dcb4)
