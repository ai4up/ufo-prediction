import re
import logging
import itertools
from collections import defaultdict
import math

import numpy as np
import matplotlib.pyplot as plt
import jellyfish
from sklearn import cluster

logger = logging.getLogger(__name__)

# arronissement is added because of inconsistencies / typo in GADM data 3.6
FRAGMENTED_CITY_REGEX = "(.*?)(-|,? +\d+er? +\(?)(Sud|Est|Ouest|Nord|arrondissement|arronissement)(-|\)| |$)"



def get_fragmented_cities_regex(gadm_boundaries, level=4):
    """
    find all cities which include Sud, Est, Ouest, Nord, Canton or arrondissement syllabus
    group them based on their basename (also includes the city with just the basename if existing)
    """
    gadm_region_columns = [f'NAME_{l}' for l in range(level + 1)]
    gadm_boundaries = gadm_boundaries.drop_duplicates(subset=gadm_region_columns)

    # get fragmented city candidates based on regex
    frag_candidates = defaultdict(list)
    for city in gadm_boundaries[gadm_region_columns[-1]].unique():
        match = re.match(FRAGMENTED_CITY_REGEX, city)
        if match is not None:
            frag_candidates[match.group(1)].append(match.string)

    # validate that fragmented city candidates are in same region
    frag_candidates_clustered_by_region = []
    for k, v in frag_candidates.items():
        gadm_boundaries_candidate = gadm_boundaries[gadm_boundaries[gadm_region_columns[-1]].isin([k] + v)]
        frag_candidates_clustered = gadm_boundaries_candidate.groupby(
            gadm_region_columns[:-1])[gadm_region_columns[-1]].apply(list).values
        frag_candidates_clustered_w_new_name = tuple((k, frag) for frag in frag_candidates_clustered)

        frag_candidates_clustered_by_region.extend(frag_candidates_clustered_w_new_name)

    fragmented_cities = [c for c in frag_candidates_clustered_by_region if len(c[1]) > 1]
    return fragmented_cities


def update_gadm_boundaries(gadm_boundaries, fragmented_cities, level=4):
    df = gadm_boundaries.copy()
    gadm_region_columns = [f'NAME_{l}' for l in range(level + 1)]

    for name, fragments in fragmented_cities:
        df.loc[df[f'NAME_{level}'].isin(fragments), f'NAME_{level}'] = name

    logger.warning(
        f'Level {level}+ attributes of first fragment will be used when aggregating fragments, rendering attributes like GID_4 misleading.')
    return df.dissolve(gadm_region_columns, aggfunc='first').reset_index()


def get_fragmented_cities_clustering(data_boundaries, level=4):
    gadm_region_columns = [f'NAME_{l}' for l in range(level + 1)]
    df = data_boundaries.drop_duplicates(subset=gadm_region_columns)
    cities_grouped_by_region = df.groupby(gadm_region_columns[:-1])[gadm_region_columns[-1]].apply(list).values

    fragmented_city_candidates = []
    for cities in cities_grouped_by_region:
        regional_candidates = _cluster_cities_based_on_string_distance(cities)
        fragmented_city_candidates.extend(regional_candidates.values())

    return fragmented_city_candidates


def _cluster_cities_based_on_string_distance(cities):
    def _lev_metric(x, y):
        i, j = int(x[0]), int(y[0])  # extract indices
        avg_length = (len(cities[i]) + len(cities[j])) / 2
        return jellyfish.levenshtein_distance(cities[i], cities[j]) / avg_length

    def _jw_metric(x, y):
        i, j = int(x[0]), int(y[0])  # extract indices
        return 1 - jellyfish.jaro_winkler(cities[i], cities[j])

    X = np.arange(len(cities)).reshape(-1, 1)
    core_samples, labels = cluster.dbscan(X, metric=_jw_metric, eps=.1, min_samples=2)

    regional_candidates = defaultdict(list)
    for i, label in enumerate(labels):
        regional_candidates[label].append(cities[i])

    # remove cities without cluster (noisy samples)
    regional_candidates.pop(-1, None)

    return regional_candidates


def visual_validation(frag_cities, boundaries_df, level=4, column_color_coding=None):
    ncols = 10 if len(frag_cities) > 10 else len(frag_cities)
    nrows = math.ceil(len(frag_cities) / 10)
    _, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(50, 20), constrained_layout=True)
    for idx, frags in enumerate(frag_cities):
        ax = axis[int(idx / 10), idx % 10] if len(frag_cities) > 10 else axis[idx % 10]
        ax.set_title(frags)
        boundaries_df[boundaries_df[f'NAME_{level}'].isin(frags)].plot(ax=ax, column=column_color_coding)


def flatten(lst):
    return list(itertools.chain(*lst))
