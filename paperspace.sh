#!/usr/bin/env bash

paperspace-python run run.py --workspace . --ignoreFiles __pycache__,.idea,.git,DS_store,datasets,assets,images --container pxeyii/maskr --machineType P100
