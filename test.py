import numpy as np

# Fungsi kernel
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Fungsi untuk menghitung nilai prediksi
def predict(w, b, X):
    return np.sign(np.dot(X, w) + b)

# Fungsi untuk melatih SVM
def train_svm(X, y, learning_rate=0.01, num_epochs=100):
    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    b = 0

    for epoch in range(num_epochs):
        for i in range(num_samples):
            if y[i] * np.dot(X[i], w) + b < 1:
                w = w + learning_rate * (y[i] * X[i] - 2 * w)
                b = b + learning_rate * y[i]

    return w, b

#
data = np.array([
    [85, 0, 85, 0, 1],
    [70, 85, 70, 72, -1],
    [89, 85, 50, 85, 1],
    [85, 89, 89, 85, -1],
    [75, 75, 72, 85, 1],
])

# Pisahkan fitur (X) dan label (y)
X = data[:, :-1]
y = data[:, -1]

# Melatih model SVM
w, b = train_svm(X, y, learning_rate=0.01, num_epochs=100)

# Data uji
X_test = np.array([
    [80, 10, 80, 10],
    [60, 70, 60, 75]
])

# Melakukan prediksi
predictions = predict(w, b, X_test)

# Menampilkan hasil prediksi
print("Hasil Prediksi:", predictions)
