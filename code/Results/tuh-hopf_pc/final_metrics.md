# Model Evaluation Report


## Task: gender

| Metric     |    Value |
|------------|----------|
| Loss (BCE) | 0.665383 |
| Accuracy   | 0.60762  |

Model prediction distribution

| Label   |   Count | %     |
|---------|---------|-------|
| label_0 |    2502 | 45.8% |
| label_1 |    2957 | 54.2% |

## Task: abnormal

| Metric     |    Value |
|------------|----------|
| Loss (BCE) | 0.482243 |
| Accuracy   | 0.768456 |

Model prediction distribution

| Label   |   Count | %     |
|---------|---------|-------|
| label_0 |    2995 | 54.9% |
| label_1 |    2464 | 45.1% |

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
