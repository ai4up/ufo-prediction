import numpy as np

AGE_ATTRIBUTE = 'age'
TYPE_ATTRIBUTE = 'type'
HEIGHT_ATTRIBUTE = 'height'
OTHER_ATTRIBUTES = [TYPE_ATTRIBUTE, HEIGHT_ATTRIBUTE, 'floors']
TARGET_ATTRIBUTES = [AGE_ATTRIBUTE, TYPE_ATTRIBUTE, HEIGHT_ATTRIBUTE, 'floors']
AUX_VARS = ['id', 'source_file', 'type_source', 'city', 'TouchesIndexes']

# OTHER_ATTRIBUTES = []
# AUX_VARS = []

# AGE_ATTRIBUTE = 'DATE_APP'
# TYPE_ATTRIBUTE = 'USAGE1'
# HEIGHT_ATTRIBUTE = 'HAUTEUR'
# OTHER_ATTRIBUTES = [TYPE_ATTRIBUTE, HEIGHT_ATTRIBUTE, 'NB_ETAGES']
# AUX_VARS = ['ID', 'USAGE2', 'PREC_ALTI', 'NB_LOGTS', 'MAT_TOITS', 'MAT_MURS',
#             'geometry', 'city', 'departement', 'is_buffer', 'TouchesIndexes']

GLOBAL_REPRODUCIBILITY_SEED = 1

# age bands ~ according to English Housing Survey (EHS) as done in https://doi.org/10.1016/j.compenvurbsys.2018.08.004
EHS_AGE_BINS = [0, 1915, 1945, 1965, 1980, 2000, np.inf]
EHS_AGE_LABELS = ['<1915', '1915-1944', '1945-1964', '1965-1979', '1980-2000', '>2000']
