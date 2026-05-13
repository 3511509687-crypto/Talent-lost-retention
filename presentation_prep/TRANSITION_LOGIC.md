# Transition Logic Between Sections

## Overall Narrative

The presentation should not sound like five isolated model explanations. The storyline is:

```text
Industrial HR attrition problem
-> convert raw employee and policy information into usable features
-> establish baseline and robustness models
-> use LightGBM for strong nonlinear prediction
-> prove the architecture through ablation and OOF Stacking
-> explain final predictions with SHAP for HR decision support
```

## Opening to Sentence-BERT

Transition:

```text
After defining the HR attrition problem, the first technical challenge is feature representation. Employee data is structured, but policy support and job context are textual. Therefore, we first need a semantic layer.
```

Logic:

- Problem asks for a real-world industrial solution.
- Real HR decisions depend on both employee records and policy environment.
- Sentence-BERT converts text into model-ready semantic features.

## Sentence-BERT to LR + ET

Transition:

```text
Once semantic and structural features are constructed, we need to test whether these features are predictive. We start with interpretable and robust baseline models before moving to the strongest nonlinear model.
```

Logic:

- Sentence-BERT creates features.
- LR tests linear separability and interpretability.
- ET tests randomized nonlinear robustness.

## LR + ET to LightGBM

Transition:

```text
The baseline models show that the features contain predictive signal, but employee attrition is unlikely to be purely linear. We therefore use LightGBM to capture stronger nonlinear interactions.
```

Logic:

- LR is interpretable but limited.
- ET is robust but not sequentially error-correcting.
- LGB provides boosting-based nonlinear learning.

## LightGBM to Ablation / OOF Stacking

Transition:

```text
Although LightGBM is strong, a single high score is not enough for an academic presentation. We need to prove which modules contribute and how the final prediction is combined without overfitting.
```

Logic:

- Strong model must be validated.
- Ablation proves module contribution.
- OOF Stacking prevents in-sample leakage in meta learning.
- Thresholding translates probabilities into decisions.

## Ablation / OOF Stacking to SHAP

Transition:

```text
After validating the model statistically, the final question is whether HR users can understand and trust the output. This is where SHAP is used.
```

Logic:

- Evaluation shows performance.
- SHAP shows reasoning.
- HR action lists require interpretability and human review.

## SHAP to Limitations

Transition:

```text
The model is useful as a decision-support prototype, but it should not be treated as a fully deployed automatic HR decision system. We therefore need to discuss limitations and future improvements.
```

Logic:

- SHAP improves transparency but does not solve causality.
- Prediction quality does not guarantee deployment readiness.
- Limitations show critical thinking and academic maturity.

## Limitations to Future Prototype

Transition:

```text
These limitations define our next development steps: external validation, probability calibration, fairness checks, and integration into a human-in-the-loop HR dashboard.
```

Logic:

- Limitations are not just weaknesses.
- They become future engineering requirements.

