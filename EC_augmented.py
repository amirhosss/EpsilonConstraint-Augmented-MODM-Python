import itertools

import numpy as np
import pyomo.environ as pyo


class Augmented():
    BETA = 1e-4

    def __init__(
        self,
        shape: tuple[int],
        step: int,
        primary_obj: int,
        obj_coefficients: list[list],
        cons_coefficients: list[list],
        cons_values: list

    ) -> None:
        self.i, self.j, self.k = shape
        self.step = step
        self.primary_obj = primary_obj
        self._obj_coefficients = np.array(obj_coefficients)
        self._cons_coefficients = np.array(cons_coefficients)
        self._cons_values = np.array(cons_values)

    @property
    def obj_coefficients(self):
        return self._obj_coefficients
    
    @obj_coefficients.setter
    def obj_coefficients(self, value):
        if len(value) != self.shape[2]:
            raise ValueError('Data shape does not match with Objective coefficients.')
        self._obj_coefficients = value 

    @property
    def cons_coefficients(self):
        return self._cons_coefficients

    @cons_coefficients.setter
    def cons_coefficients(self, value):
        if len(value) != self.shape[0]:
            raise ValueError('Data shape does not match Constraint coefficients.')
        self._cons_coefficients = value

    @property
    def cons_values(self):
        return self._cons_values

    @cons_values.setter
    def cons_values(self, value):
        if len(value) != self.shape[0]:
            raise ValueError('Data shape does not match with Constraint values.')
        self._cons_values = value

    def model_concepts(self):
        model = pyo.ConcreteModel()

        model.i = pyo.RangeSet(0, self.i-1)
        model.j = pyo.RangeSet(0, self.j-1)
        model.k = pyo.RangeSet(0, self.k-1)

        model.a = pyo.Param(model.i, initialzie=np.ndenumerate(self._cons_coefficients))
        model.c = pyo.Param(model.k, initialize=np.ndenumerate(self._obj_coefficients))
        model.b = pyo.Param(model.i, initialize=np.ndenumerate(self._cons_values))

        model.x = pyo.Var(model.j, domain=pyo.Reals)
        model.s = pyo.Var(model.k-1, domain=pyo.Reals)

        def cons_rule(model, i):
            return sum(model.a[i, j]*model.x[j] for j in model.j)
        model.cons = pyo.Constraint(model.i, rule=cons_rule)

        self.model = model
    
    def obj_function(self, coefficient: list, *x):
        coefficient = np.array(coefficient).reshape(-1, 1)
        x = np.array(x)

        return np.dot(coefficient, x)

    def calculate_income_matrix(self):
        income_matrix = np.empty((self.k, self.k))

        for m in self.k:
            self.model.obj = pyo.Objective(
                expr=sum(self.model.c[m, j]*self.model.x[j] for j in self.model.j)
            )
            opt = pyo.SolverFactory('cplex')
            opt.solve(self.model)

            for n in self.k:
                if m == n:
                    income_matrix[m, n] = pyo.value(self.model.obj)
                    continue
                x = [pyo.value(var) for var in self.model.x]
                income_matrix[m, n] = self.obj_function(self._obj_coefficients[n], x)
        
        self.income_matrix = income_matrix

    def calculate_epsilon(self):
        r_k = np.array([])
        for k in self.k:
            if k == self.primary_obj:
                continue
            indices = self.income_matrix[:, k].argsort()

            f_worst = self.income_matrix[:, indices[-2]]
            f_best = self.income_matrix[:, indices[-1]]

            r = f_worst - f_best
            r_k = np.append(r_k, r)
        
        epsilon = np.array([])
        for k in self.k:
            if k == self.primary_obj:
                continue

            eps = np.empty(self.step)
            for st in self.step:
                eps[st] = self.income_matrix[k, k] + r_k[k]*st/self.step
            np.append(epsilon, eps)

        self.epsilon_combination = itertools.product(*epsilon)

    def augmented(self):
        all_x = np.array([])
        all_f = np.array([])
        for epsilon in self.epsilon_combination:
            self.model.epsilon = pyo.Param(self.model.k-1, enumerate(epsilon))
            self.model.obj = pyo.Objective(
                expr=sum(self.model.c[self.primary_obj, j]*self.model.x[j] for j in self.model.j)
                - self.BETA*sum(self.model.s[k] for k in self.model.k-1)
            )


            def cons_rule(model, k):
                new_c = np.delete(self.model.c, self.primary_obj, 0)
                return sum(new_c[k, j]*model.x[j] for j in model.j) + model.s[k] == model.epsilon[k]
            self.model.epsilon_cons = pyo.Constraint(self.model.k-1, rule=cons_rule)

            x = np.array([pyo.value(var) for var in self.model.x])
            f = np.array([pyo.value(self.model.obj)])

            np.append(all_x, x)
            np.append(all_f, f)

        return all_f.min()