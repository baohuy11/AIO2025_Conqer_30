# Linear Regression

## **1. Gradient Descent**

### Steps for a Single Gradient Descent Update

1.  Pick a sample $(x, y)$ from training data

2.  Compute the output $\hat{y}$
    $$
    \hat{y} = wx + b
    $$

3.  Compute loss
    $$
    L = (\hat{y} - y)^2
    $$

4.  Compute derivative
    $$
    \frac{\partial L}{\partial w} = 2x(\hat{y} - y) \qquad \frac{\partial L}{\partial b} = 2(\hat{y} - y)
    $$

5.  Update parameters
    $$
    w = w - \eta \frac{\partial L}{\partial w} \qquad b = b - \eta \frac{\partial L}{\partial b}
    $$
    where $\eta$ is the learning rate.

## **2. Vetorizer**

### For a Single Sample (Stochastic Gradient Descent)

1.  Pick a sample $(\mathbf{x}, y)$ from training data

2.  Compute output $\hat{y}$
    $$
    \hat{y} = \boldsymbol{\theta}^T \mathbf{x} = \mathbf{x}^T \boldsymbol{\theta}
    $$

3.  Compute loss
    $$
    L = (\hat{y} - y)^2
    $$

4.  Compute gradient
    $$
    \nabla_{\boldsymbol{\theta}}L = 2\mathbf{x}(\hat{y} - y)
    $$

5.  Update parameters
    $$
    \boldsymbol{\theta} = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}}L
    $$
    where $\eta$ is the learning rate.

---

### For *m* Samples (Batch Gradient Descent)

1.  Pick *m* samples $(\mathbf{x}^{(i)}, y^{(i)})$ from training data

2.  Compute output $\hat{y}^{(i)}$ for each sample
    $$
    \hat{y}^{(i)} = \boldsymbol{\theta}^T \mathbf{x}^{(i)} \quad \text{for } i=1, ..., m
    $$

3.  Compute average loss over the batch
    $$
    L = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
    $$

4.  Compute the average gradient over the batch
    $$
    \nabla_{\boldsymbol{\theta}}L_{avg} = \frac{1}{m} \sum_{i=1}^{m} \nabla_{\boldsymbol{\theta}}L^{(i)}
    $$

5.  Update parameters using the average gradient
    $$
    \boldsymbol{\theta} = \boldsymbol{\theta} - \eta \left( \frac{1}{m} \sum_{i=1}^{m} \nabla_{\boldsymbol{\theta}}L^{(i)} \right)
    $$
    where $\eta$ is the learning rate.

### Code implementation

```python
import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:

	m, n = X.shape

	theta = np.zeros(n)

    for _ in range(iterations):

        predictions = X @ theta

        errors = predictions - y

        gradient = (1 / m) * (X.T @ errors)

        theta -= alpha * gradient

	return np.round(theta, 4)
```

## **3. Normal Equation**

### Using the below function to for Linear Regression

$$
\boldsymbol{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### Code implementation

```python
import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:

	X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    XtX = X.T @ X
    Xty = X.T @ y

    theta = np.linalg.inv(XtX) @ Xty
	return [round(val, 4) for val in theta.tolist()]
```

## **4. Affect of Loss Function**