# PGDL 2020 competition submission repository BrAIn - 3rd place

This is our submission repo for the PGDL 2020 competition. Our final submission is available in final_submission.zip. In the following we detail how to evaluate our 3 metrics on the public set. The public data is available [here](https://competitions.codalab.org/my/datasets/download/65b7f00a-4705-48cd-8bac-fc36021b0d69). A preprint detailing our submission will be soon made available. Code still refers to the measure as smoothness instead of variation, note that what we compute is the variation and that the smoothness is the inverse of variation, when variation is small -> signal is smooth and vice-versa.

## How to ingest and score on public data:

### 1) Ingest:

#### Usage:

```
TF_CPP_MIN_LOG_LEVEL=3 python ingestion_program/ingestion_tqdm.py {INPUT_DATA_PATH} ingestions/{SUBMISSION_NAME} ingestion_program {SUBMISSION_NAME}
```
#### Example:

```
TF_CPP_MIN_LOG_LEVEL=3 python ingestion_program/ingestion_tqdm.py ../public_data/input_data/ ingestions/ ingestion_program VPM_1/
```

### 2) Score:

#### Usage:

```
python scoring_program/score.py {REFERENCE_DATA_PATH} ingestions/{SUBMISSION_NAME} scores/{SUBMISSION_NAME}
```
#### Example:

```
python scoring_program/score.py ../public_data/reference_data/ ingestions/VPM_1/ scores/VPM_1/
```

## Scores

Given the timing constraints and the availability of the testing servers, we were only able to test our final submission on public and final, while our other attemps have been only tested on public/development. When the full data is available we would like to unify both tables. 

## Scores on the public/final of the submission 

| Measure                        | Public | Final | Task1 Public | Task 2 Public | Task 6 Final | Task 7 Final | Task 8 Final | Task 9 Final |
|--------------------------------|--------|-------|--------------|---------------|--------------|--------------|--------------|--------------|
| VPM (G=1) | 6.26   | 9.99  | 6.07         | 6.44          | 13.90        | 7.56         | 16.23        | 2.28         |

## Scores on the public/development of other attempts

| Measure                         | Public   | Development | Task1 Public | Task 2 Public | Task 4 Dev | Task 5 Dev |
|---------------------------------|----------|-------------|--------------|---------------|------------|------------|
| VR (G=11)          | 14.45    | 0.72        | 9.31         | 19.58         | 0.44       | 1.00       |
| WCV (G=1)     | **32.6** | 0.37        | 27.74        | 37.44         | 0.21       | 0.55       |
| VPM (G=80) | 11.17    | 13.04       | 4.80         | 17.54         | 15.42      | 10.66      |

## All attempts:

We tested more combinations than we present in the master branch. All other tests are available in all_tests.
