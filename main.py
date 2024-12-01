from knn_classifier import KNearestNeighbor

DATASET_FILE = "play_tennis.csv"

k = int(input("Enter the value of k: "))
distance_metric = input("Enter the distance metric ('euclidean' or 'manhattan'): ")

knn = KNearestNeighbor(k=k, distance_metric=distance_metric)

knn.train(DATASET_FILE)
knn.evaluate(DATASET_FILE, "knn_results_standard.txt", method="standard")
knn.evaluate(DATASET_FILE, "knn_results_loocv.txt", method="loocv")
