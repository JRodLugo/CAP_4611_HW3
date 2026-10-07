from pathlib import Path
import sys
import os
import json
import numpy as np
from scipy.optimize import approx_fprime

root = Path(__file__).parent
sys.path.insert(0, str(root / 'a3/code'))
os.chdir(root / 'a3/code')
from linear_models import WeightedLeastSquares, LeastSquaresBias, LeastSquaresPoly, LinearModel
from fun_obj import RobustRegressionLoss
from optimizers import GradientDescent, GradientDescentLineSearch
from utils import load_dataset

d = load_dataset('outliersData.pkl')
X, y = d['X'], d['y'].reshape(-1)
v = np.ones(len(y)); v[400:] = .1
m = WeightedLeastSquares(); m.fit(X, y, v)
assert m.predict(X).shape == y.shape
assert np.allclose(X.T @ (v * (X @ m.w-y)), 0, atol=1e-8)
loss = RobustRegressionLoss()
w = np.array([.37])
numeric = approx_fprime(w, lambda z: loss.evaluate(z, X, y)[0], 1e-6)
assert np.allclose(numeric, loss.evaluate(w, X, y)[1], atol=1e-3)
curves = {}
for name, cls in [('GD', GradientDescent), ('Line search', GradientDescentLineSearch)]:
    model = LinearModel(loss, cls(optimal_tolerance=0, max_evals=100))
    model.fit(X,y)
    assert len(model.fs) == 101 and np.isfinite(model.fs).all()
    curves[name] = [model.fs[0], model.fs[1], model.fs[5], model.fs[10], model.fs[100]]
b = load_dataset('basisData.pkl')
xb, yb = b['X'], b['y'].reshape(-1)
mb = LeastSquaresBias(); mb.fit(xb,yb)
assert np.allclose(np.column_stack((np.ones(len(xb)),xb)).T @ (mb.predict(xb)-yb), 0, atol=1e-7)
mp = LeastSquaresPoly(3); mp.fit(xb,yb)
assert np.allclose(mp._poly_basis(xb)[:,0], 1)
assert np.allclose(mp._poly_basis(xb)[:,3], xb[:,0]**3)
results = {'weighted_slope': float(m.w[0]), 'weighted_mse': float(np.mean((m.predict(X)-y)**2)), 'bias_coefficients': mb.w.tolist(), 'curves':curves}
print(json.dumps(results, indent=2))
(root / 'a3/results.json').write_text(json.dumps(results,indent=2))
print('Weighted normal equations, bias normal equations, robust gradient, basis, and 100-update curves verified.')
