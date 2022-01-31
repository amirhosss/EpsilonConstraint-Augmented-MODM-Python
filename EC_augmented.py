from itertools import product

import numpy as np
import pyomo.environ as pyo


class Augmented():
    BETA = 1e-4

    def __init__(
        self,
        shape: tuple[int],
        obj_coefficients: list[list],
        cons_coefficients: list[list],
        cons_values: list
    ) -> None:
        self.i, self.j, self.k = shape
        self._obj_coefficients = obj_coefficients
        self._cons_coefficients = cons_coefficients
        self._cons_values = cons_values

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

        model.x = pyo.Var(domain=pyo.Reals)

        def cons_rule(model, i):
            return sum(model.a[i, j]*model.x[j] for j in model.j)
        model.cons = pyo.Constraint(model.i, rule=cons_rule)