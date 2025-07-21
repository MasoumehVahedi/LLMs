import time
from finetuner import FineTuner
from data_loader import load_data
from tester import Tester


# 1) First, we need a prompt builder:
# IMPORTANT NOTE: System messages are optional.
# If our assistant follows the standard behavior (e.g., “You are a helpful assistant”), then no explicit system message is needed.
# We can safely omit it unless you need to customize behavior.
def price_prompt(item):
    system_message = "You estimate prices of items. Reply only with a number, no text or currency symbols."
    user_message = item.testPrompt(prefix="Price is $").replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Price is $"}
    ]



def main():
    # 2) Create a FineTuner instance:
    finetuner = FineTuner(wandb_project="my-fine-tuning-gpt-price-prediction")

    # ─── Step1: JSONL Dataset Preparation ─────────────────────────────────────
    train_data, test_data = load_data()
    print(price_prompt(test_data[0]))

    fine_tune_train = train_data[:500]
    fine_tune_validation = train_data[500:550]

    # 1a) Turn our items into the JSONL string
    jsonl_train = finetuner.make_jsonl(fine_tune_train, price_prompt)
    jsonl_valid = finetuner.make_jsonl(fine_tune_validation, price_prompt)

    # 1b) Write the JSONL strings to files
    finetuner.write_file("fine_tune_train.jsonl", jsonl_train)
    finetuner.write_file("fine_tune_validation.jsonl", jsonl_valid)

    # 1c) Upload the files to OpenAI
    train_file = finetuner.upload("fine_tune_train.jsonl")
    valid_file = finetuner.upload("fine_tune_validation.jsonl")

    # ─── Step2: Launch & Monitor Fine‑Tune ────────────────────────────────────
    # 2a) Create the job (fires it off)
    job = finetuner.create_fine_tune_job(
        training_file=train_file.id,
        validation_file=valid_file.id,
        model="gpt-4o-mini-2024-07-18",
        suffix="price-estimation"
    )
    # 2b) Grab the most recent job’s ID
    job_id = finetuner.list_recent_job_id()

    ######### IMPORTANT NOTE #########
    # Poll until the fine-tuning is done!
    while True:
        job_record = finetuner.retrieve_job(job_id)
        if job_record.status == "succeeded":
            break
        elif job_record.status in ["failed", "cancelled"]:
            raise Exception(f"Fine-tuning job failed with status: {job_record.status}")
        else:
            print("Job is still in progress. Waiting for 60 seconds...")
            time.sleep(60)

    # Now retrieve the final record
    fine_tuned_model_name = job_record.fine_tuned_model
    print(f"Finished! Fine-tuned model: {fine_tuned_model_name}")

    # 2c) Retrieve its current status
    job_record = finetuner.retrieve_job(job_id)
    print(f"Fine-tune job status: {job_record.status}")

    # 2d) List the training log events
    events = finetuner.list_events(job_id)
    for ev in events:
        print(f"{ev.created_at}: [{ev.level}] {ev.type} → {ev.data or ev.message}")


    # ─── Step3: Build our Predictor & Evaluate ─────────────────────────────────────
    # 3a) Build a callable that wraps our new model
    predictor = finetuner.build_predictor(
        prompt_fn=price_prompt,
        model_name=fine_tuned_model_name,
        max_tokens=5,
    )

    # True price:
    print("Truth:", test_data[0].price)
    # Our fine‑tuned model’s estimate:
    print("Pred :", predictor(test_data[0]))

    # 3b) Run it on our test set
    Tester(predictor, test_data[:250])





if __name__ == "__main__":
    main()