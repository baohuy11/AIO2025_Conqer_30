# Summary: Explainable AI (XAI) and Anchor

## **Introduction to XAI and the Need for Anchors**

**Explainable AI (XAI)** aims to make the decisions of complex "black-box" models, like neural networks, understandable to humans. An early method for this is **LIME (Local Interpretable Model-agnostic Explanations)**, which approximates a complex model's behavior around a single prediction using a simpler, linear model.

However, LIME has some drawbacks:
* **Unclear Boundaries**: It's hard to know how far the linear approximation is valid.
* **Instability**: Small changes in the data can lead to very different explanations.

The **Anchor** algorithm was developed to address these issues. Instead of approximating the model, it finds a set of **"if-then" rules** (an anchor) that are sufficient to lock in the model's prediction. The goal is to provide a clear, high-precision rule that explains why the model made a specific decision, indicating the rule's coverage for unseen cases.

---

## **Core Concepts of the Anchor Algorithm**

The Anchor method is built on a few key mathematical ideas to provide statistically robust explanations.

### **What is an Anchor?**

An **anchor** is a rule composed of one or more feature conditions (predicates). For any similar data point that satisfies this rule, the black-box model is highly likely to return the same prediction.

For example, for a loan approval model, an anchor might be:
> `IF credit_score > 750 AND income > $50k THEN PREDICT 'Approve'`

### **Precision and Coverage**

Two main metrics are used to evaluate an anchor:

1.  **Precision**: This measures how often the model's prediction remains the same for other data points that also satisfy the anchor's conditions. High precision means the rule is reliable.
    * The formula for precision is: $prec(A) = E_{D(z|A)}[1_{f(x)=f(z)}]$ 
    * This calculates the expected value that the prediction for a new sample `z` (drawn from a distribution `D` where anchor `A` holds) is the same as the original prediction for `x`.

2.  **Coverage**: This measures how often the anchor's rule applies to the overall dataset. A rule with high coverage is more general.
    * The formula for coverage is: $cov(A) = E_{D(Z)}[A(z)]$ 

The algorithm's main goal is to find an anchor that maximizes **coverage** while ensuring its **precision** is above a certain threshold, $\tau$ (e.g., 95%), with high confidence ($1-\delta$).

---

## **How the Anchor Algorithm Works**

Finding the best anchor is computationally intensive, so the algorithm uses clever search and statistical validation techniques.

### **1. Generating Candidate Anchors (Beam Search)**

The algorithm starts with an empty rule and iteratively adds feature conditions one by one in a process called **bottom-up construction**. To avoid exploring every possible combination of features, it uses **Beam Search**, which keeps only the top-k most promising rules at each step to expand upon next. This efficiently biases the search toward simple, high-precision rules.

### **2. Evaluating Candidates (Multi-Armed Bandits & KL-LUCB)**

Evaluating the true precision of each candidate rule would require too many samples. To solve this, the algorithm treats each candidate rule as an arm in a **Multi-Armed Bandit (MAB)** problem. The goal is to efficiently identify the best "arm" (the rule with the highest precision) with the fewest possible samples.

Instead of using standard statistical bounds like Hoeffding's inequality, which can be inefficient, the Anchor algorithm uses the **KL-LUCB algorithm**. KL-LUCB provides tighter, more efficient confidence bounds, reducing the number of samples needed to confirm that an anchor's precision is above the required threshold $\tau$ with high probability.

The core statistical guarantee is expressed as:
$$P(prec(A) \ge \tau) \ge 1 - \delta$$
This means: "The probability that the true precision of anchor A is at least $\tau$ is greater than or equal to $1 - \delta$."

* $\tau$: The minimum desired precision (e.g., 0.95).
* $\delta$: The probability of failure (e.g., 0.05 for 95% confidence).

---

## **Code Example: Anchor for Tabular Data**

The `alibi` library in Python provides a ready-to-use implementation of the Anchor algorithm. Here's how you can use it to explain a prediction from a Random Forest model trained on the Iris dataset. This code demonstrates how to find an anchor that explains why the model classified a specific flower as 'setosa'.

    ```python
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from alibi.explainers import AnchorTabular

    # 1. Load the Iris dataset and train a simple model
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = data.target_names

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    # 2. Set up the Anchor explainer
    # The explainer needs a function that takes raw data and returns predictions
    predict_fn = lambda x: clf.predict(x)

    # Initialize the explainer with the prediction function and feature names
    explainer = AnchorTabular(
        predictor=predict_fn,
        feature_names=feature_names
    )

    # Fit the explainer on the training data to learn data distributions
    explainer.fit(X)

    # 3. Explain a specific instance
    # Select the first instance from the dataset to explain
    instance_to_explain = X[0].reshape(1, -1)
    true_label = class_names[y[0]]

    print(f"Instance being explained: {instance_to_explain}")
    print(f"Model's prediction: {class_names[clf.predict(instance_to_explain)[0]]}")
    print(f"True class: {true_label}")

    # Generate the explanation
    explanation = explainer.explain(instance_to_explain, threshold=0.95)

    # 4. Print the results
    print("\n--- Anchor Explanation ---")
    print(f"Anchor: {' AND '.join(explanation.anchor)}")
    print(f"Precision: {explanation.precision:.2f}")
    print(f"Coverage: {explanation.coverage:.2f}")
    ```

Output of the Code:

    ```python
    Instance being explained: [[5.1 3.5 1.4 0.2]]
    Model's prediction: setosa
    True class: setosa

    --- Anchor Explanation ---
    Anchor: petal width (cm) <= 0.60 AND petal length (cm) <= 1.90
    Precision: 1.00
    Coverage: 0.33
    ```

This output tells us that for the model to classify this flower as 'setosa', it was sufficient for the petal width to be less than or equal to 0.60 cm and the petal length to be less than or equal to 1.90 cm. This rule has 100% precision, meaning that for all other samples where this rule holds, the model also predicted 'setosa'. The rule applies to about 33% of the data (coverage).
