# Prediction building age from urban form at large scale

A supervised machine learning approach to predict the construction year of buildings based on publicly available 2D urban morphology data.

## Demo
See [prediction demo](demo.ipynb) Jupyter notebook using building data from France, Spain, and the Netherlands.
![](./figures/data-overview-color-coded.png)

## Dependencies
Using pip:
```
$ pip install -r requirements.txt
```
Using conda:
```
conda env create --file=environment.yml
```

## Reproduce results
![](./figures/methods.svg)

* **Data**: Harmonized European buidling data used from https://eubucco.com/data [1].
* **Features engineering**: Urban form features crafted with [eubucco](https://github.com/ai4up/eubucco/blob/a9096afa20dc422c9063c742d66f802772ab7159/eubucco/ft_eng/ft_eng.py).
* **Experiments**: All experiments conducted to answer the research questions are defined in [bin/experiments.py](bin/experiments.py).
* **Deployment**: A Slurm cluster was utilized for model training. Submit a Slurm job to reproduce the experiments using:
    * `sbatch bin/slurm-submit/submit-prepare.sh` to prepare the data
    * `sbatch bin/slurm-submit/submit-preliminary.sh` for all preliminary experiments
    * `sbatch bin/slurm-submit/submit-exp.sh` for all main experiments
* **Figures**: All figures and tables have been created with [notebooks/v0_1_prediction.ipynb](notebooks/v0_1_prediction.ipynb) (partially outdated).


[1] Milojevic-Dupont, Nikola, and Wagner, Felix, Hu, Jiawei, Zumwald, Marius, Nachtigall, Florian, Biljecki, Filip, Heeren, Niko, Kaack, Lynn, Pichler, Peter-Paul, & Creutzig, Felix. (2022). EUBUCCO (v0.1). Zenodo. https://doi.org/10.5281/zenodo.6524781

## Abstract

> To stay within 1.5°C of global warming, reducing energy-related emissions in the building sector is essential. Rather than generic climate recommendations, this requires tailored, low-carbon urban planning solutions and spatially explicit methods that can inform policy measures at urban, street and building scale.
>
> Here, we propose a scalable method that is able to predict building age information in different countries using only open urban morphology data. 
>
> We find that spatially cross-validated regression models are sufficiently robust to generalize and predict building age in unseen cities with a mean absolute error (MAE) between 15.3 years (Netherlands) and 19.9 years (Spain).  Our experiments show that large-scale models improve generalization for predicting across cities, but are not needed to infer missing data within known cities.
>
> We further find that classification outperforms regression for use cases where only the construction period is of interest such as energy modeling.
> Overall, our results demonstrate the feasibility of generating missing age data in different contexts across Europe, providing important initial results for large-scale data generation projects such as EUBUCCO. We also highlight challenges posed by data inconsistencies and urban form differences between countries that need to be addressed for an actual roll-out of such methods.

## Contact
For any questions, please contact info@eubucco.com.
