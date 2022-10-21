import os
import sys

PROJECT_ROOT = os.path.realpath(os.path.join(__file__, '..', '..'))
PROJECT_SRC = os.path.join(PROJECT_ROOT, 'src')
SUBMODULE = os.path.join(PROJECT_ROOT, 'cluster-utils')

sys.path.append(PROJECT_SRC)
sys.path.append(SUBMODULE)
