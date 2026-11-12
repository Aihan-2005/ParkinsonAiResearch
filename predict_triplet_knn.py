"""Classify unlabeled spiral images with the CV07 triplet projection and KNN."""
from predict_contrastive_knn import main as predict_metric, predict_images as predict_metric_images


def predict_images(backbone, projection, checkpoint, paths, device):
    return predict_metric_images(backbone, projection, checkpoint, paths, device, experiment="CV07")


def main(argv=None):
    predict_metric(argv, experiment="CV07")


if __name__ == "__main__":
    main()
