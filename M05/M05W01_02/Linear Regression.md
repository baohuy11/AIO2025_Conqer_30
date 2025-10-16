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

**Normal Equation** là một phương pháp giải tích (analytical solution) để tìm ra bộ tham số tối ưu cho mô hình Linear Regression. Thay vì phải huấn luyện mô hình qua nhiều vòng lặp như Gradient Descent, Normal Equation cho phép chúng ta tính toán trực tiếp giá trị tham số tối ưu chỉ bằng một công thức duy nhất.

**Ý tưởng cốt lõi:** Thay vì "dò dẫm" tìm kiếm nghiệm tối ưu như Gradient Descent, Normal Equation giải trực tiếp phương trình đạo hàm bằng 0 để tìm điểm cực tiểu của hàm mất mát (loss function).

### **Công thức của Normal Equation của linear regression:**


$$
\boldsymbol{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

Trong đó:

* $\boldsymbol{\theta}$ (theta): Vector chứa các tham số cần tìm của mô hình

* $\mathbf{X}$: Ma trận dữ liệu đầu vào (mỗi hàng là một mẫu dữ liệu)

* $\mathbf{X}^T$: Ma trận chuyển vị của X

* $\mathbf{y}$: Vector chứa giá trị thực tế (target values)

* $(\mathbf{X}^T \mathbf{X})^{-1}$: Ma trận nghịch đảo của $(\mathbf{X}^T \mathbf{X})$

### **Cách hoạt động**

**Bước 1: Chuẩn bị dữ liệu**

Giả sử chúng ta có dữ liệu về diện tích nhà và giá nhà:

Diện tích (m²) | Giá (triệu đồng)
--- | ---
50 | 1500
80 | 2400
100 | 3000
120 | 3600

**Bước 2: Xây dựng ma trận X và vector y**

Ma trận X cần thêm cột 1 ở đầu (để tính hệ số chặn - intercept):

$$
\mathbf{X} = \begin{bmatrix}
1 & 50 \\
1 & 80 \\
1 & 100 \\
1 & 120
\end{bmatrix}
$$

Vector y:

$$
\mathbf{y} = \begin{bmatrix}
1500 \\
2400 \\
3000 \\
3600
\end{bmatrix}
$$


**Bước 3: Áp dụng công thức**

Thực hiện các phép tính ma trận theo công thức Normal Equation để tìm $\boldsymbol{\theta}$.


### **Code implementation**

```python
import numpy as np

# Data
X = np.array([[1, 50], [1, 80], [1, 100], [1, 120]])
y = np.array([[1500], [2400], [3000], [3600]])

# Apply Normal Equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# Result: theta[0] = intercept, theta[1] = slope (gradient)
print("Optimal parameters:", theta) 
```

### Ưu và nhược điểm

| ✅ Ưu điểm (Advantages) | ❌ Nhược điểm (Disadvantages) |
| :---------------------------------------------- | :------------------------------------------------------------------ |
| Không cần chọn learning rate.                  | Chậm với dữ liệu lớn (khi số features `n` > 10,000).                 |
| Không cần lặp nhiều vòng.                      | Độ phức tạp tính toán là $O(n^3)$.                                  |
| Tính toán một lần ra kết quả chính xác.         | Không hoạt động nếu ma trận $(\mathbf{X}^T \mathbf{X})$ không khả nghịch. |
| Đơn giản, dễ implement.                        | Tốn bộ nhớ để lưu các ma trận lớn.                                 |


### So sánh với Gradient Descent

#### Khi nào dùng Normal Equation?
- Khi số lượng features nhỏ (ví dụ: `n` ≤ 10,000).
- Khi cần kết quả chính xác ngay lập tức.
- Khi không muốn phải tinh chỉnh các siêu tham số (hyperparameters) như learning rate.

#### Khi nào dùng Gradient Descent?
- Khi số lượng features rất lớn (`n` > 10,000).
- Khi tập dữ liệu quá lớn, không thể tải hết vào RAM cùng một lúc.
- Khi cần một mô hình có thể học online (cập nhật khi có dữ liệu mới).



### Kết luận

**Normal Equation** là một công cụ mạnh mẽ và thanh lịch trong Machine Learning, đặc biệt phù hợp với các bài toán Linear Regression có quy mô vừa phải. Việc hiểu rõ cách hoạt động của nó không chỉ giúp bạn giải quyết vấn đề hiệu quả mà còn củng cố nền tảng toán học trong ML.

### Điểm quan trọng
**Normal Equation** là một ví dụ tuyệt vời cho thấy không phải lúc nào cũng cần "học" (learning) qua nhiều vòng lặp – đôi khi chúng ta có thể "tính toán" (computing) trực tiếp để ra ngay câu trả lời tối ưu!


## **4. Affect of Loss Function**