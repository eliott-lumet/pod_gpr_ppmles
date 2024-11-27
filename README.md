# POD-GPR surrogate modeling for miscroscale pollutant dispersion

This repository gives an example of how to build and validate a POD--GPR [1] surrogate model of microscale pollutant dispersion. As shown in [2], this surrogate can accurately learn the dependence of the 3-D mean concentration field on meteorological forcing (the inlet wind direction and friction velocity) based on a dataset of precomputed LES called PPMLES [3]. It can be used to significantly accelerate dispersion predictions and for applications that require large ensemble of model evaluations, such as data assimilation [4].

The PPMLES dataset will soon be available online at Zenodo.

### Author
Eliott Lumet

### Module requirements
- numpy
- h5py
- scikit-learn

### References

[1] Marrel, A., Perot, N., and Mottet, C. (2015). Development of a surrogate model and sensitivity analysis for spatio-temporal numerical simulators. Stochastic Environmental Research and Risk Assessment, 29(3):959–974. ISSN 1436-3259. DOI: [10.1007/s00477-014-0927-y](https://doi.org/10.1007/s00477-014-0927-y).

[2] Lumet, E., Rochoux, M. C., Jaravel, T., and Lacroix, S. (2025). Uncertainty-Aware Surrogate Modeling for Urban Air Pollutant Dispersion Prediction. Building and Environment, page 112287. DOI: [10.1016/j.buildenv.2024.112287](https://doi.org/10.1016/j.buildenv.2024.112287).

[3] Lumet, E., Jaravel, T., and Rochoux, M. C. (2024)a. PPMLES – Perturbed-Parameter ensemble of MUST Large-Eddy Simulations. Dataset. Zenodo. DOI: [10.5281/zenodo.11394347](https://doi.org/10.5281/zenodo.11394347).

[4] Lumet, E. (2024)b. Assessing and reducing uncertainty in large-eddy simulation for microscale atmospheric dispersion. PhD thesis, Université Toulouse III - Paul Sabatier. URL: [https://theses.fr/2024TLSES003](https://theses.fr/2024TLSES003). Accessed: 2024-07-08.
