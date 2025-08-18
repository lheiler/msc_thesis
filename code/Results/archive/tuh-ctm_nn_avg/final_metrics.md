# Model Evaluation Report


## Task: gender

| Metric     |    Value |
|------------|----------|
| Loss (BCE) | 0.706685 |
| Accuracy   | 0.547101 |

Model prediction distribution

| Label   |   Count | %     |
|---------|---------|-------|
| label_0 |     173 | 62.7% |
| label_1 |     103 | 37.3% |

## Task: age

| Metric     |       Value |
|------------|-------------|
| Loss (MSE) | 1.23504e-32 |
| MAE        | 1.11132e-16 |
| RMSE       | 1.11132e-16 |

## Task: abnormal

| Metric     |    Value |
|------------|----------|
| Loss (BCE) | 0.552117 |
| Accuracy   | 0.717391 |

Model prediction distribution

| Label   |   Count | %     |
|---------|---------|-------|
| label_0 |     152 | 55.1% |
| label_1 |     124 | 44.9% |

## Dataset statistics – train

|---------|------|
| Samples | 2717 |

Gender counts

| Gender code   |   Count | %     |
|---------------|---------|-------|
| female(2)     |    1447 | 53.3% |
| male(1)       |    1270 | 46.7% |
| unknown(0)    |       0 | 0.0%  |

Abnormal counts

| Label       |   Count | %     |
|-------------|---------|-------|
| abnormal(1) |    1346 | 49.5% |
| normal(0)   |    1371 | 50.5% |

Age distribution

| Age bin   |    N |
|-----------|------|
| 0–10      | 2717 |
| 10–20     |    0 |
| 20–30     |    0 |
| 30–40     |    0 |
| 40–50     |    0 |
| 50–60     |    0 |
| 60–70     |    0 |
| 70–80     |    0 |
| 80–120    |    0 |

## Dataset statistics – eval

|---------|-----|
| Samples | 276 |

Gender counts

| Gender code   |   Count | %     |
|---------------|---------|-------|
| female(2)     |     148 | 53.6% |
| male(1)       |     128 | 46.4% |
| unknown(0)    |       0 | 0.0%  |

Abnormal counts

| Label       |   Count | %     |
|-------------|---------|-------|
| abnormal(1) |     126 | 45.7% |
| normal(0)   |     150 | 54.3% |

Age distribution

| Age bin   |   N |
|-----------|-----|
| 0–10      | 276 |
| 10–20     |   0 |
| 20–30     |   0 |
| 30–40     |   0 |
| 40–50     |   0 |
| 50–60     |   0 |
| 60–70     |   0 |
| 70–80     |   0 |
| 80–120    |   0 |

## Latent-feature independence

Global HSIC score: **n/a**

(See `hsic_matrix.png` for full matrix.)
