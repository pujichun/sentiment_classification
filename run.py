from train_model import SentimentClassification
from config import model_config


def main():
    s = SentimentClassification(**model_config)
    s.fit()


if __name__ == '__main__':
    main()
