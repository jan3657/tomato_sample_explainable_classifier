# Data Availability

Raw datasets are intentionally not included in this repository until the related paper is accepted.

To run the pipeline locally:

1. Obtain the private dataset from the project authors.
2. Place it anywhere on your machine.
3. Update `data.path` in `configs/reproducible_run.yaml` to that local file.

Common local paths used during development:

- `../TomatoPredictor/Vsi_podatki_brez_null.xlsx`
- `data/tomato_elements_isotopes.csv`

Required columns:

- `Sample`
- `Target`
- feature columns used for modeling
