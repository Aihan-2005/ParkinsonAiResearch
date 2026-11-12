"""CV04: focal WideResNet features -> 64-D contrastive projection -> KNN."""
from train_representation import main as train_representation


def main(argv=None):
    train_representation(argv, cv04=True)


if __name__ == "__main__":
    main()
