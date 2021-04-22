# %cd "/content/drive/"    # Change directory to the location defined in project_path
# !git clone "https://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@github.com/iakubovskii7/MAB.git" # clone the github repository

from IPython.display import JSON
from google.colab import output
from subprocess import getoutput

def shell(command):
  if command.startswith('cd'):
    path = command.strip().split(maxsplit=1)[1]
    os.chdir(path)
    return JSON([''])
  return JSON([getoutput(command)])
output.register_callback('shell', shell)

import shutil
# shutil.rmtree("/content/drive/MyDrive/MAB/")

import os
import glob
import re
from collections import Counter
from random import choices
from typing import List, Dict
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime


os.chdir("/content/drive/Data")

