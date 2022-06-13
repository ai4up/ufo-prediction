#!/usr/bin/env python

import os
import sys

PROJECT_ROOT = os.path.realpath(os.path.join(__file__, '..', '..'))
PROJECT_SRC = os.path.join(PROJECT_ROOT, 'src')
SUBMODULE = os.path.join(PROJECT_ROOT, 'cluster-utils')

sys.path.append(PROJECT_SRC)
sys.path.append(SUBMODULE)

from prediction_age import AgePredictor
from cluster_utils import crawling


COUNTRY = 'spain'
RESULT_DIR = '/p/tmp/floriann/ml-training'
USERNAME = 'floriann'

model_path = crawling.scp_latest(pattern=f'{RESULT_DIR}/model-{COUNTRY}*.pkl', username=USERNAME)
model = AgePredictor.load(model_path)
model.evaluate()
