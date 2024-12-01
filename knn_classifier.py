import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class KNearestNeighbor:
    def __init__(self, k: int = 3, distance_metric: str = "euclidean"):
        """
        Initialize the k-NN classifier.
        :param k: Number of nearest neighbors to consider.
        :param distance_metric: Metric to use for distance calculation ("euclidean" or "manhattan").
        """
        self.k = k
        self.distance_metric = distance_metric
        self.data = None

    def train(self, dataset_path: str) -> None:
        """
        Train the k-NN classifier by loading the training dataset and storing it as a model.
        :param dataset_path: Path to the dataset in CSV format.
        """
        self.data = pd.read_csv(dataset_path)

    def _distance(self, row1: np.ndarray, row2: np.ndarray) -> float:
        """
        Compute the distance between two rows using the selected metric.
        :param row1: First row as a numpy array.
        :param row2: Second row as a numpy array.
        :return: Computed distance.
        """
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((row1 - row2) ** 2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(row1 - row2))
        else:
            raise ValueError(
                "Unsupported distance metric. Choose 'euclidean' or 'manhattan'."
            )

    def classify_instance(
        self, instance: Dict[str, any]
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Classify a single instance using the stored training dataset.
        :param instance: A dictionary representing the instance to classify.
        :return: Predicted class and distances for the k-nearest neighbors.
        """
        # Convert categorical instance features to numeric using one-hot encoding
        data_encoded = pd.get_dummies(self.data.drop(columns=["PlayTennis"]))
        instance_encoded = pd.get_dummies(pd.DataFrame([instance])).reindex(
            columns=data_encoded.columns, fill_value=0
        )

        # Calculate distances to all training examples
        distances = []
        for idx, row in data_encoded.iterrows():
            dist = self._distance(row.values, instance_encoded.values[0])
            distances.append((self.data.iloc[idx]["PlayTennis"], dist))

        # Sort distances and select k-nearest neighbors
        distances = sorted(distances, key=lambda x: x[1])
        k_nearest = distances[: self.k]

        # Predict the most frequent class among the k-nearest neighbors
        classes = [neighbor[0] for neighbor in k_nearest]
        predicted_class = max(set(classes), key=classes.count)

        return predicted_class, k_nearest

    def evaluate(self, dataset_path: str, output_file: str, method: str):
        data = pd.read_csv(dataset_path)
        correct = 0
        total_instances = len(data)

        # metrics
        tp, tn, fp, fn = 0, 0, 0, 0

        with open(output_file, "w") as f:
            f.write(f"===== k-NN Classification Results ({method}) =====\n")
            f.write(f"k: {self.k}, Distance Metric: {self.distance_metric}\n\n")
            f.write("--------------------------------------------------\n")
            f.write("| Instance |   Actual   |   Predicted  |  Correct |\n")
            f.write("--------------------------------------------------\n")

            for i in range(total_instances):
                if method == "loocv":
                    test_instance = data.iloc[i].drop("PlayTennis").to_dict()
                    actual_label = data.iloc[i]["PlayTennis"]
                    train_data = data.drop(index=i)
                    train_data.to_csv("temp_train.csv", index=False)
                    self.train("temp_train.csv")
                else:
                    test_instance = data.iloc[i].drop("PlayTennis").to_dict()
                    actual_label = data.iloc[i]["PlayTennis"]

                predicted_label, _ = self.classify_instance(test_instance)
                is_correct = predicted_label == actual_label
                correct += is_correct

                # Update confusion matrix values
                if actual_label == "Yes" and predicted_label == "Yes":
                    tp += 1
                elif actual_label == "No" and predicted_label == "No":
                    tn += 1
                elif actual_label == "No" and predicted_label == "Yes":
                    fp += 1
                elif actual_label == "Yes" and predicted_label == "No":
                    fn += 1

                # Log the result
                f.write(
                    f"|   {i+1:<5}  |   {actual_label:<8}  |   {predicted_label:<8}  |   {'True' if is_correct else 'False'}   |\n"
                )

            accuracy = correct / total_instances
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0
            )

            # Write metrics
            f.write("--------------------------------------------------\n")
            f.write(f"Overall Accuracy: {accuracy:.2f}\n\n")
            f.write("--------------------------------------------------\n")
            f.write("Confusion Matrix:\n")
            f.write("--------------------------------------------------\n")
            f.write("                   Predicted\n")
            f.write("            |   Yes    |   No     |\n")
            f.write("------------|----------|----------|\n")
            f.write(f"Actual Yes  |   {tp:<8} |   {fn:<8} |\n")
            f.write(f"Actual No   |   {fp:<8} |   {tn:<8} |\n")
            f.write("--------------------------------------------------\n\n")
            f.write(f"True Positives (TP):  {tp}\n")
            f.write(f"True Negatives (TN):  {tn}\n")
            f.write(f"False Positives (FP): {fp}\n")
            f.write(f"False Negatives (FN): {fn}\n")
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall: {recall:.2f}\n")
            f.write(f"F1 Score: {f1_score:.2f}\n")
            f.write("--------------------------------------------------\n")
