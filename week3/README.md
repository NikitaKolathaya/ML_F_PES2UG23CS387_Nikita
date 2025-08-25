# ML_F_PES2UG23CS387_Nikita
ML lab

**TICTACTOE DATASET**

<img width="554" height="399" alt="image" src="https://github.com/user-attachments/assets/e773562e-9a83-44b7-acbc-4e9eb713290d" />

Roots and early splits are usually the middle squares because they decide winning conditions.

Class distribution is balanced between win (positive) and loss(negative)

Decision patterns: Tree first checks the critical winning lines

Overfitting indicators: Highly complex tree but still generalizes (87% accuracy)


**NURSERY DATASET**

<img width="585" height="385" alt="image" src="https://github.com/user-attachments/assets/d5cec873-7b0a-4530-bd3e-e997d9c6aa21" />

Root splits have attributes like **parents*, *has_nurs*

Class Distribution is very imbalanced because most are **not_recom**, while the other values are rare.

Decision Patterns: Longer and more complex because of many classes and imbalance

Overfitting indicators: Accuracy very high, but recall is lower meaning tree struggles with minority classes. Therefore it is *overfitting* to the **not_recom** class.(Deeper branches to cover rare cases)


**MUSHROOM DATASET**

<img width="661" height="394" alt="image" src="https://github.com/user-attachments/assets/fd9b8622-7ef0-49b8-a067-0d5e711b37c5" />

Odor and spore-print-color dominate the root and erly splits.

Class Distribution: Fairly balanced

Decision Patterns: Very simple rules, single feature splits

Overfitting  indicators: Achieves 100% accuracy, not overfitting because it is seperable by a few categorical features.


**Algorithm Performance:**

**Highest Accuracy:** Mushroom dataset (100%) because features like odor are perfect predictors.

**Effect of Dataset Size:** Larger datasets (Nursery: 12960, Mushroom: 8124) are more stable and accurate models. Smaller dataset (Tictactoe: 958) leads to lower accuracy, more variance.

**Role of Features:** More features and multi-valued attributes (Nursery, Mushroom) give richer splits. Tictactoe has only 9 simple attributes, so tree is more complex to capture patterns.

**Data Characteristics Impact**

**Class Imbalance:** Nursery has imbalance because tree tends to predict not_recom often; affects macro precision/recall because of bias in predictions. Mushrooms is mostly balanced, so no bias.

**Binary vs Multi-valued:** Multi-valued categorical features (Mushroom, Nursery) help trees split cleanly and improve accuracy. Binary (Tictactoe outcomes) makes patterns harder to learn.

**Practical Applications**

**Tictactoe:** Game AI, strategy modeling, rule-based systems. Advantage: interpretable strategies (which moves lead to wins).

**Nursery:** Admission or resource allocation systems. Advantage: clear decision rules for fairness/priority.

**Mushroom:** Food safety and toxicology. Advantage: easily interpretable Eg. If odor = foul, poisonous.

**Improving Performance:**

**Tictactoe:** Use pruning or Random Forests to reduce overfitting and capture general patterns.

**Nursery:** Handle class imbalance (resampling, weighted loss).

**Mushroom:** Already perfect no improvements needed

**CONCLUSION**

Mushrooms dataset was **clean** and **seperable** 

*simple,perfect model*

Nursery dataset was **large** and **imbalanced** 

*very accurate but careful handling of minority classes*

Tictactoe dataset was **small** and **binary**   

*very complex tree lower accuracy*
