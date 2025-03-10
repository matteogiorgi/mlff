import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Calcola la distanza di x da tutti i punti di training
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Prendi gli indici dei k vicini più prossimi
        k_indices = np.argsort(distances)[:self.k]

        # Estrai le etichette corrispondenti ai k vicini
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Maggioranza delle etichette
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__"
    # Esempio di utilizzo con un dataset semplice
    dataset = np.array([
        [1, 2],
        [2, 3],
        [3, 1],
        [6, 5],
        [7, 7],
        [8, 6]
    ])

    labels = np.array([0, 0, 0, 1, 1, 1])

    # Dividiamo in dati di addestramento e test
    X_train, X_test = dataset[:5], dataset[5:]
    y_train, y_test = labels[:5], labels[5:]

    # Inizializza e allena il modello
    clf = KNN(k=3)
    clf.fit(X_train, y_train)

    # Esegui la previsione
    predictions = clf.predict(X_test)
    print("Previsioni:", predictions)
    print("Valori reali:", y_test)

    # Calcolo della precisione
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"Accuratezza: {accuracy:.2f}")
