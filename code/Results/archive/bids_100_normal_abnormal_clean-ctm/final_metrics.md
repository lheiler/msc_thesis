# Model Evaluation Report


## Task: gender

| Metric     |    Value |
|------------|----------|
| Loss (BCE) | 0.716104 |
| Accuracy   | 0.447368 |

Model prediction distribution

| Label   |   Count | %     |
|---------|---------|-------|
| label_0 |      18 | 47.4% |
| label_1 |      20 | 52.6% |

## Task: age

| Metric     |     Value |
|------------|-----------|
| Loss (MSE) | 2255.53   |
| MAE        |   39.8868 |
| RMSE       |   47.4924 |

## Task: abnormal

| Metric     |    Value |
|------------|----------|
| Loss (BCE) | 0.673162 |
| Accuracy   | 0.5      |

Model prediction distribution

| Label   |   Count | %     |
|---------|---------|-------|
| label_0 |      26 | 68.4% |
| label_1 |      12 | 31.6% |

## Dataset statistics – train

|---------|-----|
| Samples | 149 |

Gender counts

| Gender code   |   Count | %     |
|---------------|---------|-------|
| female(2)     |      78 | 52.3% |
| male(1)       |      71 | 47.7% |
| unknown(0)    |       0 | 0.0%  |

Abnormal counts

| Label       |   Count | %     |
|-------------|---------|-------|
| abnormal(1) |      74 | 49.7% |
| normal(0)   |      75 | 50.3% |

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

| Gender code   |   Count | %     |
|---------------|---------|-------|
| female(2)     |      23 | 60.5% |
| male(1)       |      15 | 39.5% |
| unknown(0)    |       0 | 0.0%  |

Abnormal counts

| Label       |   Count | %     |
|-------------|---------|-------|
| abnormal(1) |      21 | 55.3% |
| normal(0)   |      17 | 44.7% |

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

Global HSIC score: **0.012650377117097378**

(See `hsic_matrix.png` for full matrix.)
