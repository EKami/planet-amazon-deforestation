#!/bin/bash
config="$HOME/.jupyter/jupyter_notebook_config.py"
backup="$HOME/.jupyter/jupyter_notebook_config_backup.py"
cp $config $backup
cp "ipython3-versioncontrol/ipython_notebook_config.txt" "$HOME/.jupyter/"
echo "" >> "$HOME/.jupyter/ipython_notebook_config.txt" # Adds newline
cat $config >> "$HOME/.jupyter/ipython_notebook_config.txt"
rm $config
mv "$HOME/.jupyter/ipython_notebook_config.txt" $config
echo "Jupyter vc config added to jupyter default config"
