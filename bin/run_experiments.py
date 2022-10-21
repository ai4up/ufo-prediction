#!/usr/bin/env python

import os
import logging

import lib_importer
import hyperparameter_tuning
import experiments as exp
import prelim_experiments as prelim

logger = logging.getLogger(__name__)


def run_experiment():
    rq = os.environ.get('RQ')
    method = os.environ.get('METHOD')

    if rq == '0' or rq == 'prelim':
        hyperparameter_tuning.tune(method)
        prelim.model_selection(method)
        prelim.compare_resampling_strategies()
        prelim.analyze_spatial_autocorrelation()

    if rq == '1' or rq == 'regression-classification-comparison':
        exp.compare_class_vs_reg()
        exp.compare_energy_error()
    
    elif rq == '2' or rq == 'local-inference':
        exp.compare_countries(method)
        exp.evaluate_specialized_regional_models(method)
    
    elif rq == '3' or rq == 'regional-generalization':
        exp.generalize_across_countries(method)
    
    elif rq == '3b' or rq == 'spatial-distance':
        exp.evaluate_impact_of_spatial_distance_on_generalization(method, 'ESP')
        exp.evaluate_impact_of_spatial_distance_on_generalization(method, 'NLD')
        exp.evaluate_impact_of_spatial_distance_on_generalization(method, 'FRA')
    
    elif rq == '4' or rq == 'additional-data':
        exp.evaluate_impact_of_additional_data(method, exploit_spatial_autocorrelation=True, include_data_from_other_countries=True)
        exp.evaluate_impact_of_additional_data(method, exploit_spatial_autocorrelation=False, include_data_from_other_countries=True)
        exp.evaluate_impact_of_additional_data(method, across_countries=True, include_data_from_other_countries=True)
        exp.evaluate_impact_of_additional_data(method, across_cities=True, include_data_from_other_countries=True)
    
    elif rq == '5' or rq == 'SI':
        exp.compare_countries_height_prediction()
        exp.compare_countries_type_prediction()
        exp.evaluate_impact_of_additional_features(method)
    
    else:
        logger.info(f'Specified research question {rq} is not known. Aborting...')


def run_all_experiments(method):
    exp.compare_energy_error()
    exp.compare_countries(method)
    exp.generalize_across_countries(method)
    exp.evaluate_impact_of_spatial_distance_on_generalization(method, 'ESP')
    exp.evaluate_impact_of_spatial_distance_on_generalization(method, 'NLD')
    exp.evaluate_impact_of_spatial_distance_on_generalization(method, 'FRA')
    exp.evaluate_impact_of_additional_data(method, across_cities=True)
    exp.evaluate_impact_of_additional_data(method, across_countries=True)
    exp.evaluate_impact_of_additional_data(method, exploit_spatial_autocorrelation=False)
    exp.evaluate_impact_of_additional_data(method, exploit_spatial_autocorrelation=True)


if __name__ == '__main__':
    run_experiment()
