import pickle


def load_data(train_path="train.pkl", test_path="test.pkl"):
    """ load train and test dataset we build from "Efficient Loading a Large Training Data" folder. """

    with open(train_path, "rb") as f:
        train = pickle.load(f)
    with open(test_path, "rb") as f:
        test = pickle.load(f)

    return train, test


