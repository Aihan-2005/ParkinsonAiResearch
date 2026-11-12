"""Run CV07 Triplet + KNN across fold-specific focal baselines and margins."""
from train_contrastive_knn_all_folds import main as train_folds


def main(argv=None):
    train_folds(argv, cv07=True)


if __name__ == "__main__":
    main()
