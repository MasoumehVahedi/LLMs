import os
import random
import sys
from dotenv import load_dotenv
from huggingface_hub import login

from utils import list_to_dataframe
from data_loader import load_data
from testing import Tester
from features import (
    most_common_feature_keys,
    attach_raw_features,
    compute_average_weight,
    compute_average_rank,
    most_common_brands,
    get_features,
)

from baselineModels import (
    random_price,
    constant_price,
    linear_regression_model,
    linear_regression_price,
)


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
productExample_dir = os.path.join(parent_dir, "Efficient Loading a Large Training Data")
sys.path.insert(0, productExample_dir)

from product_price_prediction import ProductExample


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=True)




def main():
    train, test = load_data()
    print(train[0].prompt)
    print(train[0].price)
    print(test[0].testPrompt(prefix="Price is $"))

    random.seed(42)

    ########## 1- Build & evaluate the random model ##########
    Tester.test(random_price)

    ########## 2- Build & evaluate the constant baseline ##########
    const_price = constant_price(train)
    Tester.test(const_price)

    ########## 3- Build features & evaluate the linear regression baseline ##########
    # 1) Create a new "features" field on items, and populate it with json parsed from the details dict
    attach_raw_features(train)
    attach_raw_features(test)
    print(train[0].features.keys())

    # Look at 20 most common features in training set
    print(most_common_feature_keys(train))

    # 2) Compute defaults once
    avg_weight = compute_average_weight(train)
    avg_rank = compute_average_rank(train)

    # Investigate the brands and look at most common 40 brands
    print(most_common_brands(train))

    # 3) Build feature-function that needs the single `item` argument
    TOP_ELECTRONICS_BRANDS = [
        "hp", "dell", "lenovo",
        "samsung", "asus", "sony",
        "canon", "apple", "intel"
    ]
    feature_fn = lambda item: get_features(item, avg_weight, avg_rank, TOP_ELECTRONICS_BRANDS, PREFIX="Price is $")

    # 4) Build DataFrames for our features
    df_train = list_to_dataframe(train, feature_fn)
    df_test = list_to_dataframe(test[:250], feature_fn)

    # 5) Linear regression function to predict price for a new item
    model, feature_columns = linear_regression_model(df_train, df_test, target="price")
    lr_pricer = linear_regression_price(model, feature_columns, feature_fn)
    Tester.test(lr_pricer)






if __name__ == "__main__":
    main()