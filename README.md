ML-engineer-agent automatically loads a dataset, writes the Python code to prepare the data and build a model, evaluates on the relevant metrics and iterates until finding the best solution.

It supports regression and classification tasks.

----

In your terminal, run:

```python main.py --data .../data/housing.csv --task regression --target median_house_value --iterations 5 --output results```

- `--data`: path to your csv
- `--task`: regression or classification
- `--target`: the target variable
- `--iterations`: number of iterations to run
- `--output`: the folder in which you want to store the iteration outputs and the best model code


```Initializing AI Agent for regression task...
Loading data from .../data/housing.csv
Starting optimization process for 3 iterations...
MSE: 3354991560.466231, R² Score: 0.7439737046849305
New best model found! Score: 0.7440```
