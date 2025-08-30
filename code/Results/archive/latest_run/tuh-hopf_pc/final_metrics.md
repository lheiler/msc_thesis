# Model Evaluation Report


## Task: abnormal

| Metric     |    Value |
|------------|----------|
| Loss (BCE) | 0.476618 |
| Accuracy   | 0.771936 |

Model prediction distribution

| Label   |   Count | %     |
|---------|---------|-------|
| label_0 |    3008 | 55.1% |
| label_1 |    2451 | 44.9% |

## Dataset statistics – train

|---------|-------|
| Samples | 53622 |

Gender counts

| Gender code   |   Count | %     |
|---------------|---------|-------|
| female(2)     |       0 | 0.0%  |
| male(1)       |   28583 | 53.3% |
| unknown(0)    |   25039 | 46.7% |

Abnormal counts

| Label       |   Count | %     |
|-------------|---------|-------|
| abnormal(1) |   26545 | 49.5% |
| normal(0)   |   27077 | 50.5% |

Age distribution

| Age bin   |     N |
|-----------|-------|
| 0–10      | 53622 |
| 10–20     |     0 |
| 20–30     |     0 |
| 30–40     |     0 |
| 40–50     |     0 |
| 50–60     |     0 |
| 60–70     |     0 |
| 70–80     |     0 |
| 80–120    |     0 |

## Dataset statistics – eval

|---------|------|
| Samples | 5459 |

Gender counts

| Gender code   |   Count | %     |
|---------------|---------|-------|
| female(2)     |       0 | 0.0%  |
| male(1)       |    2919 | 53.5% |
| unknown(0)    |    2540 | 46.5% |

Abnormal counts

| Label       |   Count | %     |
|-------------|---------|-------|
| abnormal(1) |    2486 | 45.5% |
| normal(0)   |    2973 | 54.5% |

Age distribution

| Age bin   |    N |
|-----------|------|
| 0–10      | 5459 |
| 10–20     |    0 |
| 20–30     |    0 |
| 30–40     |    0 |
| 40–50     |    0 |
| 50–60     |    0 |
| 60–70     |    0 |
| 70–80     |    0 |
| 80–120    |    0 |

## Latent-feature independence

Global HSIC score: **n/a**

(See `hsic_matrix.png` for full matrix.)
