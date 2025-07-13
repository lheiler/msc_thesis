# Model Evaluation Report


## Losses

| Metric                   |       Value |
|--------------------------|-------------|
| Loss – gender (BCE)      |    0.698117 |
| Loss – age (MSE)         | 2342.58     |
| Loss – abnormality (BCE) |    0.690186 |
| Total loss               | 2343.97     |

## Classification metrics

| Metric            |    Value |
|-------------------|----------|
| Gender accuracy   | 0.394737 |
| Abnormal accuracy | 0.552632 |
| Gender precision  | 0        |
| Gender recall     | 0        |
| Gender F1         | 0        |
| Abn precision     | 0.552632 |
| Abn recall        | 1        |
| Abn F1            | 0.711864 |

## Confusion matrices

### Gender

|        |   Pred 0 |   Pred 1 |
|--------|----------|----------|
| True 0 |       15 |        0 |
| True 1 |       23 |        0 |

### Abnormality

|        |   Pred 0 |   Pred 1 |
|--------|----------|----------|
| True 0 |        0 |       17 |
| True 1 |        0 |       21 |

## Age regression

| Metric       |     Value |
|--------------|-----------|
| MAE (years)  | 1008.55   |
| RMSE (years) |   26.4102 |

### MAE per age bin

| Age bin   |      MAE |
|-----------|----------|
| 0–20      | 37.1712  |
| 20–40     | 21.0869  |
| 40–60     |  5.70437 |
| 60–80     | 20.7072  |
| 80–100    | 37.6104  |

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

Global HSIC score: **0.013779190368950367**

(See `hsic_matrix.png` for full matrix.)
