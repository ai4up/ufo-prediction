import re
import itertools
from collections import defaultdict
import math

import numpy as np
import matplotlib.pyplot as plt
import jellyfish
from sklearn import cluster

FRAGMENTED_CITY_REGEX = "(.*?)(-| +\d+er? +\(?)(Sud|Est|Ouest|Nord|Canton|arrondissement)(-|\)| |$)"

# find all cities which include Sud, Est, Ouest, Nord or Canton syllabus
# group them based on their basename (also includes the city with just the basename if existing)
def get_fragmented_level_4_cities_regex(data_gadm_boundaries):
    # get fragmented city candidates based on regex
    fragmented_city_candidates = defaultdict(list)
    for city in data_gadm_boundaries['NAME_4'].unique():
        match = re.match(FRAGMENTED_CITY_REGEX, city)
        if match is not None:
            fragmented_city_candidates[match.group(1)].append(match.string)

    # validate that fragmented city candidates are in same region
    fragmented_city_candidates_clustered_by_region = []
    for k, v in fragmented_city_candidates.items():
        fragmented_city_candidates_clustered_by_region.extend(data_gadm_boundaries[data_gadm_boundaries['NAME_4'].isin([k]+v)].groupby('NAME_3')['NAME_4'].apply(list).values)
    fragmented_cities = [c for c in fragmented_city_candidates_clustered_by_region if len(c) > 1]

    return fragmented_cities


# def get_fragmented_level_4_cities_clustering(data_boundaries):
def get_fragmented_cities_clustering(data_boundaries, level=4):
    gadm_region_columns = [f'NAME_{l}' for l in range(level+1)]
    df = data_boundaries.drop_duplicates(subset=gadm_region_columns)
    cities_grouped_by_region = df.groupby(gadm_region_columns[:-1])[gadm_region_columns[-1]].apply(list).values

    fragmented_city_candidates = []
    for cities in cities_grouped_by_region:
        regional_candidates = _cluster_cities_based_on_string_distance(cities)
        fragmented_city_candidates.extend(regional_candidates.values())

    return fragmented_city_candidates


def get_fragmented_level_3_cities_clustering(data_boundaries):
    df = data_boundaries.drop_duplicates(subset=['NAME_0', 'NAME_1', 'NAME_2', 'NAME_3'])
    cities_grouped_by_region = df.groupby(['NAME_0', 'NAME_1', 'NAME_2'])['NAME_3'].apply(list).values
    fragmented_city_candidates = []
    for cities in cities_grouped_by_region:
        regional_candidates = _cluster_cities_based_on_string_distance(cities)
        fragmented_city_candidates.extend(regional_candidates.values())

    return fragmented_city_candidates


def _cluster_cities_based_on_string_distance(cities):
    def _lev_metric(x, y):
            i, j = int(x[0]), int(y[0]) # extract indices
            avg_length = (len(cities[i]) + len(cities[j])) / 2
            return jellyfish.levenshtein_distance(cities[i], cities[j]) / avg_length

    def _jw_metric(x, y):
            i, j = int(x[0]), int(y[0]) # extract indices
            return 1 - jellyfish.jaro_winkler(cities[i], cities[j])

    X =  np.arange(len(cities)).reshape(-1, 1)
    core_samples, labels = cluster.dbscan(X, metric=_jw_metric, eps=.1, min_samples=2)

    regional_candidates = defaultdict(list)
    for i, label in enumerate(labels):
        regional_candidates[label].append(cities[i])

    # remove cities without cluster (noisy samples)
    regional_candidates.pop(-1, None)

    return regional_candidates


def visual_validation(frag_cities, boundaries_df, level=4):
    ncols = 10 if len(frag_cities) > 10 else len(frag_cities)
    nrows = math.ceil(len(frag_cities) / 10)
    _, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(50, 20), constrained_layout=True)
    for idx, frags in enumerate(frag_cities):
        ax = axis[int(idx / 10), idx % 10] if len(frag_cities) > 10 else axis[idx % 10]
        ax.set_title(frags)
        boundaries_df[boundaries_df[f'NAME_{level}'].isin(frags)].plot(ax=ax)


def flatten(lst):
    return list(itertools.chain(*lst))
