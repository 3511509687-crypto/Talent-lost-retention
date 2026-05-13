# Presentation Preparation Pack

本目录把本次 pre 拆成 5 个技术板块，每个板块都对应一个文件夹：

1. `01_sentence_bert`: Sentence-BERT 语义表示与政策匹配
2. `02_lr_et`: Logistic Regression + ExtraTrees 基准与稳健性对照
3. `03_lightgbm`: LightGBM 核心非线性预测
4. `04_ablation_oof_stacking`: 消融实验、OOF Stacking、阈值和分箱校准
5. `05_shap`: SHAP 解释与 HR 决策支持

根目录里的全局文件负责把 5 个板块串起来：

- `PDF_REQUIREMENTS_AND_FIVE_BLOCK_CHECKLIST.md`: PDF rubric 对齐、五个技术板块准备重点、15 分钟时间分配、额外材料 checklist
- `DETAILED_PRESENTATION_OUTLINE.md`: 15 分钟 pre 的详细结构
- `LIMITATIONS_AND_DEFENSE.md`: general limitations、两个重点 limitation、每板块易被质疑点
- `REFERENCE_LIST.md`: 模型、算法、工具、评估指标对应参考文献
- `TECHNICAL_METHODS_AND_TOOLS.md`: 建模过程中用到的技术手段、语言、库、绘图工具及依据
- `TRANSITION_LOGIC.md`: 板块与板块之间的衔接话术
- `FORMULA_SYMBOL_EXPLANATIONS.md`: 五个板块中所有公式符号和未知数解释；Word 版见 `00_Formula_Symbol_Explanations.docx`

建议使用顺序：

1. 先读 `PDF_REQUIREMENTS_AND_FIVE_BLOCK_CHECKLIST.md`，确认 rubric、五板块重点和时间分配。
2. 再读 `DETAILED_PRESENTATION_OUTLINE.md`，确定整体叙事。
3. 每个讲者读自己板块的 `README.md`，挑 2 到 4 个核心图表。
4. 统一看 `LIMITATIONS_AND_DEFENSE.md`，准备 Q&A。
5. 最后把 `REFERENCE_LIST.md` 和 `TECHNICAL_METHODS_AND_TOOLS.md` 变成 references 与 methodology backup slides。
