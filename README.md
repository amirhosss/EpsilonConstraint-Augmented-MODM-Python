# Getting Started
First we create an object from __Augmented__ class and initialize "shape of problem", "steps" and "primary object".

```py
augmented = Augmented(shape=shape, step=step, primary_obj=primary_obj)
```
## Input data
Then we should initialize Constraint coefficients, Objective coefficients and Constraint values.
```py
augmented.cons_coefficients = cons_matrix
augmented.obj_coefficients = obj_matrix
augmented.cons_values = cons_val_matrix
```
## Run
Finally after declaring all variables and initializing model data, we use __run__ method as follows:
```py
augmented.run(output=False, version='v1')
```
It takes 2 arguments "output" and "version".
Augmented version (v1, v2) specified by version and X array specified by output.
## PLot
Plotting objectives by their order.
```py
augmented.plot_objectives((objective_ord, objective_ord))
```