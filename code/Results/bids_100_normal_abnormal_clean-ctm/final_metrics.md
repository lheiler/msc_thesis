# Model Evaluation Report


## Losses

| Metric                   | Value   |
|--------------------------|---------|
| Loss – gender (BCE)      |         |
| Loss – age (MSE)         |         |
| Loss – abnormality (BCE) |         |
| Total loss               |         |

## Classification metrics

| Metric            | Value   |
|-------------------|---------|
| Gender accuracy   |         |
| Abnormal accuracy |         |
| Gender precision  |         |
| Gender recall     |         |
| Gender F1         |         |
| Abn precision     |         |
| Abn recall        |         |
| Abn F1            |         |

## Confusion matrices


## Age regression

| Metric       | Value   |
|--------------|---------|
| MAE (years)  |         |
| RMSE (years) |         |

## Dataset statistics – train

|---------|-----|
| Samples | 149 |

Gender counts

| Gender code   |   Count |
|---------------|---------|
| female(2)     |      78 |
| male(1)       |      71 |
| unknown(0)    |       0 |

Abnormal counts

| Label       |   Count |
|-------------|---------|
| abnormal(1) |      74 |
| normal(0)   |      75 |

Age distribution

| Age bin   |   N |
|-----------|-----|
| 0–10      |  12 |
| 10–20     |  11 |
| 20–30     |  22 |
| 30–40     |  12 |
| 40–50     |  19 |
| 50–60     |  15 |
| 60–70     |  22 |
| 70–80     |  24 |
| 80–120    |  12 |

## Dataset statistics – eval

|---------|----|
| Samples | 38 |

Gender counts

| Gender code   |   Count |
|---------------|---------|
| female(2)     |      23 |
| male(1)       |      15 |
| unknown(0)    |       0 |

Abnormal counts

| Label       |   Count |
|-------------|---------|
| abnormal(1) |      21 |
| normal(0)   |      17 |

Age distribution

| Age bin   |   N |
|-----------|-----|
| 0–10      |   4 |
| 10–20     |   4 |
| 20–30     |  11 |
| 30–40     |   1 |
| 40–50     |   2 |
| 50–60     |   4 |
| 60–70     |   6 |
| 70–80     |   2 |
| 80–120    |   4 |

## Latent-feature independence

Global HSIC score: **0.013779191300272942**

(See `hsic_matrix.png` for full matrix.)
