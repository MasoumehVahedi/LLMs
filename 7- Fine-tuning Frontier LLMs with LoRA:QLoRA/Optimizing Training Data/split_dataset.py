import random
import pickle
from datasets import Dataset, DatasetDict



def splitDataset(sample, train_size: int, test_size: int, seed: int = 42):
    """ Break down our data into a training, test and validation dataset.
        It's typical to use 5%-10% of your data for testing purposes.
    """
    random.seed(seed)
    random.shuffle(sample)
    x_train = sample[:train_size]
    x_test = sample[train_size: train_size + test_size]
    print(f"Divided into a training set of {len(x_train):,} records and test set of {len(x_test):,} records.")

    # Serialize raw objects so we can re-load quickly next time
    with open("train.pkl", "wb") as file:
        pickle.dump(x_train, file)

    with open("test.pkl", "wb") as file:
        pickle.dump(x_test, file)

    return x_train, x_test



def buildDataset(x_train, x_test, PREFIX: str):
    """ Convert train and test data to prompts and upload to HuggingFace hub. """

    train_prompts = [record.prompt for record in x_train]
    train_prices = [record.price for record in x_train]
    test_prompts = [record.testPrompt(prefix=PREFIX) for record in x_test]
    test_prices = [record.price for record in x_test]

    # Create a Dataset from the lists
    train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
    test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Upload our brand new dataset to HuggingFace Hub
    # HF_USER = "MY USERNAME"
    # DATASET_NAME = f"{HF_USER}/pricer-data"
    # dataset.push_to_hub(DATASET_NAME, private=True)

    return dataset
