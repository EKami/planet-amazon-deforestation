#!/bin/bash
path="$HOME/.jupyter/jupyter_notebook_config.py"
backup="$HOME/.jupyter/jupyter_notebook_config_backup.py"
cp $backup $path
echo "Backup restored from $backup"
