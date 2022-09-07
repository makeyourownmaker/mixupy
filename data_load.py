
import numpy as np
import pandas as pd
from mixupy import mixup
# from mixupy.mixup import mixup

cam = pd.read_csv('../CambridgeTemperatureNotebooks/data/CamMetCleanish2021.04.26.csv', nrows=1000)
cam.drop('ds', axis=1, inplace=True)
