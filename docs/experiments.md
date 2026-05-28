# Experiments

## Prelims
1. Tiered threshold sweep
2. Clipping sweep
3. Minor hyperparam sweep

## Planned
1. Change the training/test ratios to 0.9:0.1, 0.85:0.15, and 0.95:0.05.
2. Include all three types of NRMSE in the model evaluation:
   - Mean
   - Range
   - std-based
3. Plot actual vs. predicted values after running the model to evaluate how closely the predictions match the observed values.
4. Analyze the relationships among NRMSE, COV, %Error using training graphs.
   - Explain what each graph line represents for each metric
   - Summarize the conclusions from the trends and correlations
5. Find the optimal hyperparameters using Random Search and Grid Search with proper justification for the selected parameters.
6. Categorize dist_coast and dist_reservoir into distance ranges (e.g., 5-10 km intervals.).
7. Check feature selection approaches for variables such as age, number of stories, and property value, including comparisons between sum- and average-based features.
