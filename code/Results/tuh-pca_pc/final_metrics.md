# Model Evaluation Report


## Task: gender

| Metric     |    Value |
|------------|----------|
| Loss (BCE) | 0.655047 |
| Accuracy   | 0.604506 |

Model prediction distribution

| Label   |   Count | %     |
|---------|---------|-------|
| label_0 |    3261 | 59.7% |
| label_1 |    2198 | 40.3% |

## Task: abnormal

| Metric     |    Value |
|------------|----------|
| Loss (BCE) | 0.481848 |
| Accuracy   | 0.772303 |

Model prediction distribution

| Label   |   Count | %     |
|---------|---------|-------|
| label_0 |    2978 | 54.6% |
| label_1 |    2481 | 45.4% |

## Dataset statistics – train

|---------|-------|
| Samples | 53622 |

Gender counts

| Gender code   |   Count | %     |
|---------------|---------|-------|
| female(0)     |   25039 | 46.7% |
| male(1)       |   28583 | 53.3% |

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
| female(0)     |    2540 | 46.5% |
| male(1)       |    2919 | 53.5% |

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
