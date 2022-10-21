#!/usr/bin/env python

import os
import sys

import lib_importer
from prediction_age import AgePredictor
from cluster_utils import crawling


COUNTRY = 'spain'
RESULT_DIR = '/p/tmp/floriann/ml-training'
USERNAME = 'floriann'

model_path = crawling.scp_latest(pattern=f'{RESULT_DIR}/model-{COUNTRY}*.pkl', username=USERNAME)
model = AgePredictor.load(model_path)
model.evaluate()
