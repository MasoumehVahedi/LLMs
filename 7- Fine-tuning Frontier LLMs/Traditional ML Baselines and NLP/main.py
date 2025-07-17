import os
import random
import sys
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR

from utils import list_to_dataframe
from utils import export_human_input, load_human_predictions, make_human_price
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
    bow_linear_regression_model,
    w2v_regression_model
)

from frontierModels import make_frontier_pricer, prepare_messages


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
    Tester.test(random_price, test)

    ########## 2- Build & evaluate the constant baseline ##########
    const_price = constant_price(train)
    Tester.test(const_price, test)

    ########## 3- Build features & evaluate the linear regression baseline ##########
    # 1) Create a new "features" field on items, and populate it with json parsed from the details dict
    attach_raw_features(train)
    attach_raw_features(test)
    print(train[0].features.keys())

    # Look at 20 most common features in training set
    # This will help us understand what features we can use. One feature is "Brand", another is "Item Weight", etc.
    print(most_common_feature_keys(train))

    # 2) Compute defaults once
    avg_weight = compute_average_weight(train)
    avg_rank = compute_average_rank(train)

    # Investigate the brands and look at most common 40 brands
    print(most_common_brands(train))

    # 3) Build feature-function that needs the single `item` argument
    TOP_BRANDS = [
        "Fender", "Frigidaire", "GE",
        "Whirlpool", "Rockville", "Pyle",
        "Generic", "Ibanez", "Behringer"
    ]
    feature_fn = lambda item: get_features(item, avg_weight, avg_rank, TOP_BRANDS, PREFIX="Price is $")

    # 4) Build DataFrames for our features
    df_train = list_to_dataframe(train, feature_fn)
    df_test = list_to_dataframe(test[:250], feature_fn)

    # 5) Linear regression function to predict price for a new item
    model, feature_columns = linear_regression_model(df_train, df_test, target="price")
    lr_pricer = linear_regression_price(model, feature_columns, feature_fn)
    Tester.test(lr_pricer, test)

    ########## 4- Build Bag of Words + LinearRegression baseline ##########
    # Extract training docs & targets
    prices = np.array([float(item.price) for item in train])
    documents = [item.testPrompt(prefix="Price is $") for item in train]

    # Train and evaluate our BOW+LR baseline
    bow_lr_price = bow_linear_regression_model(documents, prices)
    Tester.test(bow_lr_price, test)

    ########## 5- Build Word2Vec + LinearRegression baseline ##########
    # Build your Word2Vec+LR pricer
    w2v_price = w2v_regression_model(documents,
                                     prices,
                                     regressor_cls=LinearRegression,
                                     regressor_kwargs={})
    Tester.test(w2v_price, test)

    ########## 6- Build Word2Vec + LinearSVR baseline ##########
    svr_price = w2v_regression_model(documents,
                                     prices,
                                     regressor_cls=LinearSVR,
                                     regressor_kwargs={"random_state":42, "max_iter":10000})
    Tester.test(svr_price, test)


    ########## 7- Build Word2Vec + Random Forest baseline ##########
    rf_price = w2v_regression_model(documents,
                                     prices,
                                     regressor_cls=RandomForestRegressor,
                                     regressor_kwargs={"n_estimators":100, "random_state":42, "n_jobs":8})
    Tester.test(rf_price, test)


    ########## 8- Build human price baseline ##########
    # Export prompts for humans
    export_human_input(test, filename="human_input.csv")

    # Load human estimates
    human_preds = load_human_predictions("human_output.csv")

    # Build the human-pricer and evaluate
    human_pricer = make_human_price(test, human_preds)
    Tester.test(human_pricer, test)

    ########## 9- Build Frontier model: GPT ##########
    print(prepare_messages(test[0]))
    print(test[0].price)
    # Use the Frontier model to estimate prices
    frontier_gpt4o_price = make_frontier_pricer("gpt-4o-mini")
    Tester.test(frontier_gpt4o_price, test)

    # The function for gpt-4o - the August model
    gpt4o_xl_price = make_frontier_pricer("gpt-4o-2024-08-06", max_tokens=5, temperature=0.2)
    Tester.test(gpt4o_xl_price, test)



if __name__ == "__main__":
    main()