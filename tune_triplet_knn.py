"""Select CV07 triplet and KNN settings using development folds only."""
from tune_contrastive_knn import main as select_metric


def main(argv=None):
    select_metric(argv, experiment="CV07")


if __name__ == "__main__":
    main()
