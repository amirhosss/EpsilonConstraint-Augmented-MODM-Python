import itertools
# from typing import Optional

import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt


class Augmented():
    MAX_PLOT_OBJ = 2
    BETA = 1e-4

    def __init__(
        self,
        shape: tuple[int],
        step: int,
        primary_obj: int

    ) -> None:
        self.i, self.j, self.k = shape
        self.step = step
        self.primary_obj = primary_obj
        self._obj_coefficients = None
        self._cons_coefficients = None
        self._cons_values = None

    @property
    def obj_coefficients(self):
        return self._obj_coefficients
    
    @obj_coefficients.setter
    def obj_coefficients(self, value):
        if len(value) != self.k:
            raise ValueError('Data shape does not match with Objective coefficients.')
        self._obj_coefficients = value 

    @property
    def cons_coefficients(self):
        return self._cons_coefficients

    @cons_coefficients.setter
    def cons_coefficients(self, value):
        if len(value) != self.i:
            raise ValueError('Data shape does not match Constraint coefficients.')
        self._cons_coefficients = value

    @property
    def cons_values(self):
        return self._cons_values

    @cons_values.setter
    def cons_values(self, value):
        if len(value) != self.i:
            raise ValueError('Data shape does not match with Constraint values.')
        self._cons_values = value

    def model_concepts(self) -> None:
        model = pyo.ConcreteModel()

        model.i = pyo.RangeSet(0, self.i-1)
        model.j = pyo.RangeSet(0, self.j-1)
        model.k = pyo.RangeSet(0, self.k-1)
        model.z = pyo.RangeSet(0, self.k-2)

        model.a = pyo.Param(model.i, model.j, initialize=dict(np.ndenumerate(self._cons_coefficients)))
        model.c = pyo.Param(model.k, model.j, initialize=dict(np.ndenumerate(self._obj_coefficients)))
        model.b = pyo.Param(model.i, initialize=dict(np.ndenumerate(self._cons_values)))

        model.x = pyo.Var(model.j, domain=pyo.NonNegativeReals)
        model.s = pyo.Var(pyo.RangeSet(0, self.k-2), domain=pyo.NonNegativeReals)

        def cons_rule(model: pyo.ConcreteModel, i: int):
            return sum(model.a[i, j]*model.x[j] for j in model.j) <= model.b[i]
        model.cons = pyo.Constraint(model.i, rule=cons_rule)

        self.model = model
    
    def obj_function(self, coefficient: list, *x) -> np.ndarray:
        coefficient = np.array(coefficient).reshape(-1, 1)
        x = np.array(x)

        return np.dot(x, coefficient)

    def calculate_income_matrix(self) -> None:
        income_matrix = np.empty((self.k, self.k))

        for m in range(self.k):
            self.model.obj = pyo.Objective(
                expr=sum(self.model.c[m, j]*self.model.x[j] for j in self.model.j)
            )
            opt = pyo.SolverFactory('cplex')
            opt.solve(self.model)

            for n in range(self.k):
                if m == n:
                    income_matrix[m, n] = pyo.value(self.model.obj)
                    continue
                x = [pyo.value(self.model.x[j]) for j in self.model.j]
                income_matrix[m, n] = self.obj_function(self._obj_coefficients[n], *x)

            self.model.del_component(self.model.obj)
        
        self.income_matrix = income_matrix

    def calculate_epsilon(self) -> None:
        r_k = {}
        for k in range(self.k):
            if k == self.primary_obj:
                continue

            indices = self.income_matrix[:, k].argsort()

            f_worst = self.income_matrix[:, k][indices[-1]]
            f_best = self.income_matrix[:, k][indices[0]]

            r = f_worst - f_best
            r_k[k] = r

        self.r_k = r_k
        
        epsilon = {}
        for k in range(self.k):
            if k == self.primary_obj:
                continue

            eps = np.empty(self.step+1)
            for st in range(self.step+1):
                eps[st] = self.income_matrix[k, k] + r_k[k]*st/self.step
            
            epsilon[k] = eps

        self.epsilon_combination = np.array(list(itertools.product(*list(epsilon.values()))))

    def augmented(self, version: str ='v1') -> None:
        if not version in ['v1', 'v2']:
            raise ValueError('Version is not correct.')

        all_x = []
        obj_versions ={
            'v1': pyo.Objective(
                expr=sum(self.model.c[self.primary_obj, j]*self.model.x[j] for j in self.model.j)
                - self.BETA*sum(self.model.s[k]/list(self.r_k.keys())[k] for k in self.model.z)
            ),
            'v2': pyo.Objective(
                expr=sum(self.model.c[self.primary_obj, j]*self.model.x[j] for j in self.model.j)
                - self.BETA*sum(self.model.s[k]/list(self.r_k.keys())[k] for k in self.model.z)
            )
        }

        for epsilon in self.epsilon_combination:
            self.model.epsilon = pyo.Param(self.model.z, initialize=dict(enumerate(epsilon)))
            self.model.obj = obj_versions[version]

            def cons_rule(model: pyo.ConcreteModel, k: int):
                new_c = np.delete(self.model.c, self.primary_obj, 0)
                return sum(new_c[k, j]*model.x[j] for j in model.j) + model.s[k] == model.epsilon[k]
            self.model.epsilon_cons = pyo.Constraint(self.model.z, rule=cons_rule)

            opt = pyo.SolverFactory('cplex')
            opt.solve(self.model)
            
            x = np.array([pyo.value(self.model.x[j]) for j in self.model.j])
            all_x.append(x)

            self.model.del_component(self.model.epsilon)
            self.model.del_component(self.model.obj)
            self.model.del_component(self.model.epsilon_cons)

        self.all_x = all_x

    def plot_objectives(self, objectives: tuple[int]) -> None:
        if len(objectives) != self.MAX_PLOT_OBJ:
            raise ValueError('Wrong objectives tuple')
        all_objs = [
            [self.obj_function(self._obj_coefficients[objectives[0]], *x) for x in self.all_x],
            [self.obj_function(self._obj_coefficients[objectives[1]], *x) for x in self.all_x]
        ]
        plt.plot(*all_objs, 'bo')
        plt.xlabel('Objective function '+str(objectives[0]))
        plt.ylabel('Objective function '+str(objectives[1]))
        plt.grid()
        plt.show()


    def run(self, output=False, version='v1') -> None:
        self.model_concepts()
        self.calculate_income_matrix()
        self.calculate_epsilon()
        self.augmented(version=version)
        
        if output:
            print(self.all_x) 