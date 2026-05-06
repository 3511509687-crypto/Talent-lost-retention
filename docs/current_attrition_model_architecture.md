# Current Attrition Model Architecture

This version is intentionally simplified for presentation clarity and follows a left-to-right reading order.

Key code anchors:

- `run_pipeline`: [models/v3_1_blue.py](F:/app_bundle/models/v3_1_blue.py#L3754)
- `train_stacking_lgb`: [models/v3_1_blue.py](F:/app_bundle/models/v3_1_blue.py#L2421)
- `train_multi_seed_ensemble`: [models/v3_1_blue.py](F:/app_bundle/models/v3_1_blue.py#L2831)
- `BlendedAttritionModel`: [models/v3_1_blue.py](F:/app_bundle/models/v3_1_blue.py#L2210)
- `MultiSeedEnsembleModel`: [models/v3_1_blue.py](F:/app_bundle/models/v3_1_blue.py#L2242)

## Mermaid Source

```mermaid
flowchart LR
    classDef input fill:#eef6ff,stroke:#5b88c7,color:#15324f,stroke-width:1.4px
    classDef feature fill:#eefdf5,stroke:#46a56f,color:#12392a,stroke-width:1.4px
    classDef merge fill:#fff8e9,stroke:#d69c35,color:#49300f,stroke-width:1.4px
    classDef decision fill:#fff1f2,stroke:#c45f67,color:#5a1d21,stroke-width:1.4px
    classDef optional fill:#f8f3ff,stroke:#8a73d6,color:#362456,stroke-width:1.4px,stroke-dasharray: 6 4

    A["1. Input Data<br/>employee dataset + policy dataset"]:::input
    B["2A. Employee Feature Layer<br/>raw HR fields + engineered attrition signals"]:::feature
    C["2B. Policy Semantic Layer<br/>embedding + matching + policy feature outputs"]:::feature
    D["3. Merge + Preprocess<br/>merged table + ColumnTransformer"]:::merge
    E["4. Ensemble Stack<br/>LGB + LR + ET<br/>OOF blend + meta LR"]:::merge
    F["5. Decision + Outputs<br/>final probability + segment thresholds + reports"]:::decision
    G["Optional: MultiSeedEnsemble<br/>average meta probabilities across seeds"]:::optional

    A --> B
    A --> C
    B --> D
    C --> D
    D --> E --> F
    G -.-> F
```

## Reading Notes

- Default path: one stacking model from `train_stacking_lgb(...)`.
- Optional path: `enable_seed_stability=True` wraps several full stacks with `MultiSeedEnsembleModel`.
- Final classification is segment-aware, not just one global threshold.
