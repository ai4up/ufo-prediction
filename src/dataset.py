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


# EHS_AGE_BINS = [0, 1950, np.inf]
# EHS_AGE_LABELS = ['<1950', '>1950']

# EHS_AGE_BINS = [0, 1915, 1960, np.inf]
# EHS_AGE_LABELS = ['<1915', '1915-1960', '>1960']

SELECTED_FEATURES = [
    'total_ft_area_within_buffer_100',
    'street_closeness_global_closest_street',
    'StdBlockFootprintArea',
    'street_width_av_closest_street',
    'std_convexity_within_buffer_100',
    'av_convexity_within_buffer_100',
    'lon',
    'Phi',
    'street_betweeness_global_closest_street',
    'Convexity',
    'BlockLongestAxisLength',
    'street_width_std_within_buffer_500',
    'street_width_std_closest_street',
    'std_block_av_footprint_area_within_buffer_100',
    'distance_to_closest_street',
    'total_ft_area_within_buffer_500',
    'AvBlockFootprintArea',
    'lat',
    'BlockElongation',
    'BlockTotalFootprintArea',
    'street_betweeness_global_av_within_buffer_100',
    'Corners',
    'FootprintArea',
    'std_block_footprint_area_within_buffer_100',
    'buildings_within_buffer_100',
    'buildings_within_buffer_500',
    'av_footprint_area_within_buffer_100',
    'av_elongation_within_buffer_100'
    ]

# SELECTED_FEATURES = ['std_elongation_within_buffer_100',
#        'street_betweeness_global_closest_street',
#        'distance_to_closest_street', 'Convexity',
#        'std_elongation_within_buffer_500', 'buildings_within_buffer_100',
#        'lat', 'SharedWallLength', 'std_footprint_area_within_buffer_100',
#        'StdBlockFootprintArea']