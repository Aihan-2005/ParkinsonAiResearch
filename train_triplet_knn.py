"""CV07: focal WideResNet frozen features -> thesis triplet projection -> KNN."""
from train_representation import main as train_metric


def main(argv=None):
    train_metric(argv, cv07=True)


if __name__ == "__main__":
    main()
