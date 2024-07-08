# POD-GPR surrogate modeling for miscroscale pollutant dispersion

This repository gives an example of how to build and validate a POD--GPR [1] surrogate model of microscale pollutant dispersion. This surrogate learns the dependence of the
3-D mean concentration field on meteorological forcing (the inlet wind direction and friction velocity) based on a dataset of precomputed LES called PPMLES. It can be used to significantly accelerate dispersion predictions and 
for applications that require large ensemble of model evaluations, such as data assimilation [2].

The PPMLES dataset will soon be available online at Zenodo.

### References

[1] Marrel, A., Perot, N., and Mottet, C. (2015). Development of a surrogate model and sensitivity analysis for spatio-temporal numerical simulators. Stochastic Environmental Research and Risk Assessment, 29(3):959–974. ISSN 1436-3259. DOI: https://doi.org/10.1007/s00477-014-0927-y.

[2] Lumet, E. (2024). Assessing and reducing uncertainty in large-eddy simulation for microscale atmospheric dispersion. PhD thesis, Université Toulouse III - Paul Sabatier. URL: https://theses.fr/2024TLSES003. Accessed: 2024-07-08.
