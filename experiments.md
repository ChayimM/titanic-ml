# Titanic — Experiment Log

This document records the main hypotheses, experiments, model comparisons,
and conclusions developed during my Titanic machine-learning project.

The purpose is not only to maximize predictive accuracy, but to document
the reasoning process: what was tested, why it was tested, what failed,
and what conclusions can reasonably be drawn from the results.

I've obviously taken care in not falling for overfitting, rather I've put
more focus into finding useful takeaways and interesting patterns.

# 1. Baseline Model

## Objective

Establish a reproducible machine-learning baseline before experimenting
with more complex models or additional features.

## Features

The initial feature set consisted of:

- `Pclass`
- `Sex`
- `Age`
- `Fare`
- `Title`
- `FamilySize`
- `IsAlone`
- `IsMother`

## Preprocessing

A scikit-learn `Pipeline` and `ColumnTransformer` were used to keep
preprocessing inside the modelling workflow.

### Numerical features

- Median imputation for missing values

### Categorical features

- One-hot encoding
- `handle_unknown="ignore"`

## Model

Logistic Regression.

## Validation

Stratified 5-fold cross-validation was used initially, followed by
repeated stratified cross-validation for more robust comparisons.

## Results

| Metric | Result |
|---|---:|
| Cross-validation accuracy | ~0.82 |
| Kaggle accuracy | ~0.76 |

## Observation

The Kaggle score was noticeably lower than the local cross-validation
estimate.

This demonstrated that local validation performance is not necessarily
representative of performance on an unseen test set.

## Next question

Can feature engineering, model selection, and hyperparameter optimization
improve generalization?





# 2. Feature Engineering

Several features were created to capture information that is not directly
represented by the original Titanic variables.

## Title

Passenger titles were extracted from `Name`.

Examples include:

- Mr
- Miss
- Mrs
- Master

The feature was motivated by the possibility that titles capture both
age and social characteristics.

## FamilySize

Defined as:

`SibSp + Parch + 1`

This was intended to represent the number of immediate family members
travelling with the passenger.

## IsAlone

A binary feature indicating whether the passenger was travelling without
immediate family.

## IsMother

A heuristic feature identifying adult female passengers travelling with
children and/or parents, excluding passengers with the title `Miss`.

## TicketGroupSize

The number of passengers sharing the same ticket.

This feature was investigated separately because passengers sharing a
ticket may have travelled together and therefore potentially experienced
correlated outcomes.

NOTE: Later on I have found a ~0.75 association between TicketGroupSize and FamilySize.
At this point of the development process the difference was just theoretical, though.





# 3. TicketGroupSize Experiment

## Hypothesis

Passengers sharing a ticket may have correlated survival outcomes because
they may have travelled together.

## Results

| Model | Original | + TicketGroupSize |
|---|---:|---:|
| Logistic Regression | 0.819 ± 0.023 | 0.819 ± 0.023 |
| Random Forest | 0.813 ± 0.023 | 0.813 ± 0.023 |

## Interpretation

Adding `TicketGroupSize` did not improve repeated cross-validation
performance for either model.

This suggests that the information contained in ticket group size was
either:

1. not strongly useful for prediction, or
2. already represented indirectly by existing variables such as
   `FamilySize`, `Pclass`, `Sex` and perhaps most importantly `IsAlone`.

The result also demonstrates that a feature can appear intuitively
meaningful without improving predictive performance.





# 4. Group Size and Survival — Statistical Investigation

## Hypothesis

The original hypothesis was:

> Larger groups may have higher survival rates because passengers
> travelling together could potentially help one another.

However, group size is confounded by other characteristics. In particular,
large groups may be disproportionately composed of third-class passengers.

Therefore, group size cannot be evaluated responsibly without examining
its relationship with class and sex.

## 4.1 Relationship Between Candidate Group Measures

Correlation matrix:

| | TicketGroupSize | FamilySize | IsAlone | Pclass | Fare |
|---|---:|---:|---:|---:|---:|
| TicketGroupSize | 1.000  | 0.748  | -0.458 | -0.003 | 0.346  |
| FamilySize      | 0.748  | 1.000  | -0.691 | 0.066  | 0.217  |
| IsAlone         | -0.458 | -0.691 | 1.000  | 0.135  | -0.272 |
| Pclass          | -0.003 | 0.066  | 0.135  | 1.000  | -0.550 |
| Fare            | 0.346  | 0.217  | -0.272 | -0.550 | 1.000  |

`TicketGroupSize` and `FamilySize` are strongly correlated (r ≈ 0.75),
but they are not identical measures.

This motivated investigating both measures.

## 4.2 Survival by Family Size

| Family Size | Survival | Count |
|---:|---:|---:|
| 1  | 30.4% | 537 |
| 2  | 55.3% | 161 |
| 3  | 57.8% | 102 |
| 4  | 72.4% | 29  |
| 5  | 20.0% | 15  |
| 6  | 13.6% | 22  |
| 7  | 33.3% | 12  |
| 8  | 0.0%  | 6   |
| 11 | 0.0%  | 7   |

The strongest and most reliable difference occurs between travelling alone
and travelling with at least one family member.

After family sizes of approximately 4, sample sizes become very small,
making individual survival rates unreliable.

## 4.3 Survival by Ticket Group Size

| Ticket Group Size | Survival | Count |
|---:|---:|---:|
| 1 | 29.8% | 547 |
| 2 | 57.4% | 188 |
| 3 | 69.8% | 63  |
| 4 | 50.0% | 44  |
| 5 | 0.0%  | 10  |
| 6 | 0.0%  | 18  |
| 7 | 23.8% | 21  |

Again, the strongest difference occurs between being alone and travelling
with others.

Very large groups have low observed survival rates, but their sample
sizes are considerably smaller.





# 5. Group Categories

To avoid over-interpreting noisy individual group sizes, passengers were
classified into:

- `Alone`: 1
- `Small`: 2–4
- `Large`: 5+

## Family Group

| Group | Survival | Count |
|---|---:|---:|
| Alone | 30.4% | 537 |
| Small | 57.9% | 292 |
| Large | 16.1% | 62 |

## Ticket Group

| Group | Survival | Count |
|---|---:|---:|
| Alone | 29.8% | 547 |
| Small | 59.0% | 295 |
| Large | 10.2% | 49 |

## Interpretation

There is no evidence that progressively larger groups have progressively
higher survival.

Instead, the strongest pattern is:

> **Alone → substantially worse survival**  
> **Small group → substantially better survival**  
> **Large group → substantially worse survival**

However, this raw relationship cannot be interpreted as an independent
effect of group size because group size is associated with other passenger
characteristics.

# 6. Group Size, Class and Sex

## Family Group × Class

| Group | Class | Survival | Count |
|---|---:|---:|---:|
| Alone | 1 | 53.2%  | 109 |
| Alone | 2 | 34.6%  | 104 |
| Alone | 3 | 21.3%  | 324 |
| Small | 1 | 73.3%  | 101 |
| Small | 2 | 62.8%  | 78 |
| Small | 3 | 40.7%  | 113 |
| Large | 1 | 66.7%  | 6 |
| Large | 2 | 100.0% | 2 |
| Large | 3 | 7.4%   | 54 |

The large-group result is dominated by third-class passengers, with only
8 observations from first and second class combined.

This is a strong warning against interpreting the raw large-group survival
rate without considering class.

## Family Group × Sex

| Group | Sex | Survival | Count |
|---|---|---:|---:|
| Alone | Female | 78.6% | 126 |
| Alone | Male   | 15.6% | 411 |
| Small | Female | 80.6% | 155 |
| Small | Male   | 32.1% | 137 |
| Large | Female | 27.3% | 33 |
| Large | Male   | 3.4% | 29 |

The relationship between group status and survival differs substantially
by sex.

Women had high survival rates when travelling alone or in small groups,
while men showed a much larger difference between travelling alone and
travelling with others.

The large-group female result should be treated cautiously because of
the relatively small sample. It would be interesting testing this with
other datasets, to see if the pattern of large female groups having
quite low survival rates holds up. Again, however, other variables
must not be neglected.





# 7. IsAlone — Statistical Test

Because the largest and most reliable difference appeared to be between
travelling alone and travelling with others, this relationship was
examined more formally.

## Results

|          | Not Alone | Alone |
|---|---:|---:|
| Survival | **50.6%**  | **30.4%** |
| 95% CI   | 45.4–55.7% | 26.6–34.4% |

Observed difference:

**+20.2 percentage points**

The confidence intervals are sufficiently separated to indicate a
substantial difference in observed survival between the two groups.

However, this should not be interpreted as evidence that being alone
causes lower survival. Passenger sex, class, age, and other characteristics
are potential confounders.

# 8. IsAlone × Sex

| Sex | Not Alone | Alone | Difference |
|---|---:|---:|---:|
| Female | 71.3% | 78.6% | -7.3 pp |
| Male   | 27.1% | 15.6% | +11.5 pp |

Interestingly, the relationship reverses for women.

Women travelling alone had slightly higher observed survival than women
travelling with others, while men had substantially higher survival when
travelling with others.

This demonstrates why aggregate relationships can be misleading when
important subgroups behave differently.

One can come up with many different reasons, as to why it seems beneficial
as a woman to travel alone (in this specific case, at least). I have not
done any further investigation on this however.





# 9. IsAlone × Sex × Class

The strongest observed relationships involved the interaction between
sex, passenger class, and travelling status.

For example:

- First-class women had survival rates above 95% regardless of whether
  they travelled alone.
- Third-class women had substantially lower survival.
- Men had substantially lower survival than women across classes.
- Among men, both passenger class and travelling status were associated
  with survival.

These results suggest that survival was strongly structured by sex and
class, with group status providing additional information whose effect
varied between subgroups.

The analysis does **not** establish a causal explanation for these
differences.

Sex    Pclass IsAlone   Survival Rate
female 1      0        0.966667     60
              1        0.970588     34
       2      0        0.931818     44
              1        0.906250     32
       3      0        0.416667     84
              1        0.616667     60
male   1      0        0.425532     47
              1        0.333333     75
       2      0        0.277778     36
              1        0.097222     72
       3      0        0.180723     83
              1        0.121212    264





# 10. Hypothesis Conclusion

The original hypothesis that:

> "Larger travelling groups are associated with higher survival"

was **not supported**.

The observed relationship was instead non-monotonic.

Passengers travelling in small groups (2–4) had substantially higher
survival than passengers travelling alone, while very large groups had
lower observed survival.

However, the large-group effect was heavily confounded by passenger class,
and the relationship also differed by sex.

Therefore, the most defensible conclusion is:

> **Travelling status is strongly associated with survival, but the
> relationship is not a simple "larger group = higher survival" effect.
> The strongest distinction is between travelling alone and travelling
> with a small group, while large groups show substantially lower survival.
> These relationships are strongly intertwined with sex and passenger
> class.**

The investigation therefore rejected the original simple hypothesis while
revealing a more nuanced relationship worth incorporating into model
development.





# 11. Model Comparison

Three model families were investigated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

## Initial repeated cross-validation

| Model | Mean Accuracy | Std |
|---|---:|---:|
| Logistic Regression   | 0.819 | 0.023 |
| Random Forest         | 0.813 | 0.023 |
| Gradient Boosting     | 0.821 | 0.021 |

The differences were relatively small.

Logistic Regression provided a strong baseline despite being substantially
simpler than the tree-based approaches.

Random Forest did not outperform Logistic Regression on this dataset.





# 12. Gradient Boosting Hyperparameter Tuning

## Motivation

Gradient Boosting showed slightly stronger initial performance, so its
hyperparameters were investigated systematically.

The main parameters tested were:

- `learning_rate`
- `max_iter`

Twenty combinations were evaluated using `GridSearchCV`.

## Manual experiment

The strongest manually tested configuration was:

- `learning_rate = 0.05`
- `max_iter = 100`
- CV accuracy = 0.835 ± 0.021

## Grid Search

The strongest GridSearch configuration was:

- `learning_rate = 0.10`
- `max_iter = 50`
- CV accuracy = 0.847
- CV standard deviation = 0.017

The GridSearch result is a model-selection score rather than an unbiased
final performance estimate because the same cross-validation procedure was
used to select the hyperparameters.





# 13. Tuned Gradient Boosting — Robust Evaluation

The selected Gradient Boosting model was evaluated again using repeated
stratified 5-fold cross-validation.

Results:

| Metric | Result |
|---|---:|
| Mean accuracy         | **0.834** |
| Standard deviation    | **0.020** |
| Minimum               | 0.792 |
| Maximum               | 0.876 |

The tuned model therefore improved substantially over the initial
Gradient Boosting result in repeated cross-validation.

The improvement from approximately 0.821 to 0.834 is much more credible
than the 0.894 score obtained on a single validation split.

The 0.894 result illustrates the danger of relying on a single validation
split after repeatedly making modelling decisions.





# 14. Current Model Conclusion

At the current stage, tuned Gradient Boosting is the strongest model
tested.

However, the advantage over Logistic Regression remains relatively small.

This is an important result in itself:

> Greater model complexity does not automatically produce substantially
> better predictive performance.

The final model should therefore be selected based on both predictive
performance and the purpose of the project.





# 15. Lessons Learned

This project's development was guided by a complete machine-learning workflow:

1. Load and inspect data
2. Explore variables
3. Engineer features
4. Handle missing values
5. Encode categorical variables
6. Build preprocessing pipelines
7. Train multiple model families
8. Evaluate using cross-validation
9. Form and test hypotheses
10. Investigate confounding variables
11. Perform statistical analysis
12. Tune hyperparameters
13. Generate predictions
14. Evaluate against an unseen Kaggle test set
15. Document results and limitations

What I enjoyed most was hypothesizing about relationships in the data and then testing whether the evidence actually supported them. Through this process, I learned a great deal about what makes a good analysis: questioning my assumptions, testing them systematically, and avoiding conclusions that the data does not support.

Overview of what a reliable analysis requires:

- sensible feature construction,
- careful validation,
- awareness of confounding,
- appropriate statistical reasoning,
- comparison of simple and complex models,
- and honest documentation of uncertainty and failed hypotheses.

# 16. Kaggle Submission
Local cross-validation substantially overestimated performance on the Kaggle test set. This discrepancy was already present in the Logistic Regression baseline and therefore does not appear to be unique to Gradient Boosting. The result highlights the distinction between cross-validation estimates and performance on a particular unseen sample.
Tuned Gradient Boosting achieved the strongest repeated cross-validation performance among the evaluated models (0.834 ± 0.020), and was therefore selected as the final candidate. However, its Kaggle score of 0.768 was substantially lower than the cross-validation estimate, demonstrating that model selection based solely on local validation can be unreliable.

I then tried a few different seeds which gave the following results:
 Seed  CV Score  Learning Rate  Max Iter  Predicted Survival
    1  0.836156           0.02       200            0.358852
    7  0.848446           0.01       200            0.368421
   42  0.847348           0.10        50            0.361244
  123  0.835020           0.01       200            0.368421
  999  0.841755           0.02       100            0.366029