import numpy as np
import itertools
GLOBAL_REPRODUCIBILITY_SEED = 1

# age bands ~ according to English Housing Survey (EHS) as done in https://doi.org/10.1016/j.compenvurbsys.2018.08.004
EHS_AGE_BINS = [0, 1915, 1945, 1965, 1980, 2000, np.inf]
CUSTOM_AGE_BINS = [0, 1915, 1960, np.inf]

BUILDING_TYPES = [
    'Résidentiel',
    'Annexe',
    'Agricole',
    'Commercial et services',
    'Industriel',
    'Religieux',
    'Sportif',
    # Indifférencié == n.a.
]

AGE_ATTRIBUTE = 'age'
TYPE_ATTRIBUTE = 'type_source'
HEIGHT_ATTRIBUTE = 'height'
AUX_VARS = [
    'id',
    'id_source',
    'source_file',
    'city',
    'TouchesIndexes',
    'type',
    'type_source',
    'age_wsf',
    'block',
    'block_bld_ids',
    'sbb',
    'sbb_bld_ids',
    'geometry',
    'country',
]
BUILDING_FEATURES = [
    'FootprintArea',
    'Perimeter',
    'Phi',
    'LongestAxisLength',
    'Elongation',
    'Convexity',
    'Orientation',
    'Corners',
    'CountTouches',
    'SharedWallLength',
    'lat',
    'lon',
]
BUILDING_FEATURES_NEIGHBORHOOD = [
    'av_convexity_within_buffer_100',
    'av_convexity_within_buffer_500',
    'av_elongation_within_buffer_100',
    'av_elongation_within_buffer_500',
    'av_footprint_area_within_buffer_100',
    'av_footprint_area_within_buffer_500',
    'av_orientation_within_buffer_100',
    'av_orientation_within_buffer_500',
    'buildings_within_buffer_100',
    'buildings_within_buffer_500',
    'std_convexity_within_buffer_100',
    'std_convexity_within_buffer_500',
    'std_elongation_within_buffer_100',
    'std_elongation_within_buffer_500',
    'std_footprint_area_within_buffer_100',
    'std_footprint_area_within_buffer_500',
    'std_orientation_within_buffer_100',
    'std_orientation_within_buffer_500',
    'total_ft_area_within_buffer_100',
    'total_ft_area_within_buffer_500',
]
BLOCK_FEATURES = [
    'AvBlockFootprintArea',
    'BlockConvexity',
    'BlockCorners',
    'BlockElongation',
    'BlockLength',
    'BlockLongestAxisLength',
    'BlockOrientation',
    'BlockPerimeter',
    'BlockTotalFootprintArea',
    'StdBlockFootprintArea',
]
BLOCK_FEATURES_NEIGHBORHOOD = [
    'blocks_within_buffer_100',
    'blocks_within_buffer_500',
    'av_block_av_footprint_area_within_buffer_100',
    'av_block_av_footprint_area_within_buffer_500',
    'av_block_footprint_area_within_buffer_100',
    'av_block_footprint_area_within_buffer_500',
    'av_block_length_within_buffer_100',
    'av_block_length_within_buffer_500',
    'av_block_orientation_within_buffer_100',
    'av_block_orientation_within_buffer_500',
    'std_block_av_footprint_area_within_buffer_100',
    'std_block_av_footprint_area_within_buffer_500',
    'std_block_footprint_area_within_buffer_100',
    'std_block_footprint_area_within_buffer_500',
    'std_block_length_within_buffer_100',
    'std_block_length_within_buffer_500',
    'std_block_orientation_within_buffer_100',
    'std_block_orientation_within_buffer_500',
]
STREET_FEATURES = [
    'dist_to_closest_int',
    'distance_to_closest_street',
    'street_based_block_area',
    'street_based_block_corners',
    'street_based_block_phi',
    'street_length_closest_street',
    'street_openness_closest_street',
    'street_width_av_closest_street',
    'street_width_std_closest_street',
]
STREET_FEATURES_NEIGHBORHOOD = [
    'intersection_count_within_100',
    'intersection_count_within_500',
    'street_based_block_av_area_inter_buffer_100',
    'street_based_block_av_area_inter_buffer_500',
    'street_based_block_av_phi_inter_buffer_100',
    'street_based_block_av_phi_inter_buffer_500',
    'street_based_block_number_inter_buffer_100',
    'street_based_block_number_inter_buffer_500',
    'street_based_block_std_area_inter_buffer_100',
    'street_based_block_std_area_inter_buffer_500',
    'street_based_block_std_orientation_inter_buffer_100',
    'street_based_block_std_orientation_inter_buffer_500',
    'street_based_block_std_phi_inter_buffer_100',
    'street_based_block_std_phi_inter_buffer_500',
    'street_length_av_within_buffer_100',
    'street_length_av_within_buffer_500',
    'street_length_std_within_buffer_100',
    'street_length_std_within_buffer_500',
    'street_length_total_within_buffer_100',
    'street_length_total_within_buffer_500',
    'street_width_av_within_buffer_100',
    'street_width_av_within_buffer_500',
    'street_width_std_within_buffer_100',
    'street_width_std_within_buffer_500'
]
STREET_FEATURES_CENTRALITY = [
    'street_betweeness_global_av_within_buffer_100',
    'street_betweeness_global_av_within_buffer_500',
    'street_betweeness_global_closest_street',
    'street_betweeness_global_max_within_buffer_100',
    'street_betweeness_global_max_within_buffer_500',
    'street_closeness_500_av_within_buffer_100',
    'street_closeness_500_av_within_buffer_500',
    'street_closeness_500_closest_street',
    'street_closeness_500_max_within_buffer_100',
    'street_closeness_500_max_within_buffer_500',
    'street_closeness_global_closest_street',
]
CITY_FEATURES = [
    'total_buildings_city',
    'total_buildings_footprint_city',
    'av_building_footprint_city',
    'std_building_footprint_city',
    'n_detached_buildings',
    'blocks_2_to_4',
    'blocks_5_to_9',
    'blocks_10_to_19',
    'blocks_20_to_inf',
    'intersections_count',
    'total_length_street_city',
    'av_length_street_city',
    'total_number_block_city',
    'av_area_block_city',
    'std_area_block_city',
]
LANDUSE_FEATURES = [
    'bld_in_lu_agricultural',
    'bld_in_lu_industrial_commercial',
    'bld_in_lu_natural',
    'bld_in_lu_other',
    'bld_in_lu_roads',
    'bld_in_lu_urban_fabric',
    'bld_in_lu_urban_green',
    'bld_in_lu_water',
    'bld_in_lu_ocean_country',
    'bld_in_lu_railways',
    'bld_in_lu_ports_airports',
    'lu_ocean_country_within_buffer_100',
    'lu_natural_within_buffer_100',
    'lu_industrial_commercial_within_buffer_100',
    'lu_other_within_buffer_100',
    'lu_water_within_buffer_100',
    'lu_urban_green_within_buffer_100',
    'lu_agricultural_within_buffer_100',
    'lu_railways_within_buffer_100',
    'lu_urban_fabric_within_buffer_100',
    'lu_ports_airports_within_buffer_100',
    'lu_roads_within_buffer_100',
    'lu_ocean_country_within_buffer_500',
    'lu_natural_within_buffer_500',
    'lu_industrial_commercial_within_buffer_500',
    'lu_other_within_buffer_500',
    'lu_water_within_buffer_500',
    'lu_urban_green_within_buffer_500',
    'lu_agricultural_within_buffer_500',
    'lu_railways_within_buffer_500',
    'lu_urban_fabric_within_buffer_500',
    'lu_ports_airports_within_buffer_500',
    'lu_roads_within_buffer_500'
]
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
TARGET_ATTRIBUTES = [AGE_ATTRIBUTE, TYPE_ATTRIBUTE, HEIGHT_ATTRIBUTE, 'floors']

BUILDING_FEATURES_ALL = BUILDING_FEATURES + BUILDING_FEATURES_NEIGHBORHOOD
BLOCK_FEATURES_ALL = BLOCK_FEATURES + BLOCK_FEATURES_NEIGHBORHOOD
STREET_FEATURES_ALL = STREET_FEATURES + STREET_FEATURES_NEIGHBORHOOD + STREET_FEATURES_CENTRALITY

FEATURES = list(itertools.chain(
    BUILDING_FEATURES,
    BUILDING_FEATURES_NEIGHBORHOOD,
    BLOCK_FEATURES,
    BLOCK_FEATURES_NEIGHBORHOOD,
    STREET_FEATURES,
    STREET_FEATURES_NEIGHBORHOOD,
    STREET_FEATURES_CENTRALITY,
    CITY_FEATURES,
    LANDUSE_FEATURES,
))

# Top 10
# SELECTED_FEATURES = ['std_elongation_within_buffer_100',
#        'street_betweeness_global_closest_street',
#        'distance_to_closest_street', 'Convexity',
#        'std_elongation_within_buffer_500', 'buildings_within_buffer_100',
#        'lat', 'SharedWallLength', 'std_footprint_area_within_buffer_100',
#        'StdBlockFootprintArea']

# OTHER_ATTRIBUTES = []
# AUX_VARS = []

# AGE_ATTRIBUTE = 'DATE_APP'
# TYPE_ATTRIBUTE = 'USAGE1'
# HEIGHT_ATTRIBUTE = 'HAUTEUR'
# OTHER_ATTRIBUTES = [TYPE_ATTRIBUTE, HEIGHT_ATTRIBUTE, 'NB_ETAGES']
# AUX_VARS = ['ID', 'USAGE2', 'PREC_ALTI', 'NB_LOGTS', 'MAT_TOITS', 'MAT_MURS',
#             'geometry', 'city', 'departement', 'is_buffer', 'TouchesIndexes']
