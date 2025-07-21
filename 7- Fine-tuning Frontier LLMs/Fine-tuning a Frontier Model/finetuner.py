# finetuner.py
# -----------------
# Helpers to prepare data, launch a fine‑tune, and then run the resulting model.

"""
   How to set Weights & Biases (W&B) up:

    1- Create a free account
        • Go to https://wandb.ai and sign up (or log in).
        • Click our Avatar→Settings, then scroll down to “API Keys” and copy our key.

    2- Link our API key in OpenAI
        • Visit https://platform.openai.com/account/organization and open the Integrations tab.
        • Paste our W&B API key there under Weights & Biases.

    Enable it in code
    In our FineTuner (or wherever we kick off fine‑tuning), pass our project name:


"""


import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Any, Callable


# Initialize client with our API key
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)


class FineTuner:
    """
        A general class to prepare data, launch, monitor, and evaluate
        any OpenAI chat‑based fine‑tune job.
    """

    def __init__(self, wandb_project=None):
        """
            Args:
                api_key      : our OPENAI_API_KEY (default from env)
                wandb_project: optional Weight and Bias project name for job integration
        """
        self.wandb_integration = (
            [{"type": "wandb", "wandb": {"project": wandb_project}}]
            if wandb_project else []
        )


    @staticmethod
    def make_jsonl(items, prompt_fn) -> str:
        """Pack `items` into a JSONL string of {'messages':[...]}."""
        lines = []
        for obj in items:
            messages = prompt_fn(obj)
            lines.append(json.dumps({"messages": messages}))
        return "\n".join(lines)


    def write_file(self, path: str, content: str) -> None:
        """Write content to a file."""
        with open(path, "w") as f:
            f.write(content)


    def upload(self, file_path: str) -> Any:
        """Upload a JSONL file and return its OpenAI file ID."""
        with open(file_path, "rb") as f:
            response = openai_client.files.create(
                file=f,
                purpose="fine-tune",
            )
        return response


    def prepare_and_upload(self, items, prompt_fn, path):
        """
            Convert `items` → JSONL via `prompt_fn`, write to `path`, upload,
            and return the file ID.
        """
        jl = self.make_jsonl(items, prompt_fn)
        self.write_file(path, jl)
        return self.upload(path)


    def create_fine_tune_job(self, training_file: str,
                             validation_file: str,
                             model: str,
                             n_epochs: int = 1,
                             seed: int = 42,
                             suffix: str = "") -> Any:
        """
           Launch a fine‑tune and return the job ID.
        """
        params = dict(
            training_file=training_file,
            validation_file=validation_file,
            model=model,
            seed=seed,
            #integrations=[wandb_integration],
            hyperparameters={"n_epochs": n_epochs},
            suffix=suffix
        )
        if self.wandb_integration:
            params["integrations"] = self.wandb_integration

        job = openai_client.fine_tuning.jobs.create(**params)
        return job


    def list_recent_job_id(self):
        """
            Return the most‑recent fine‑tune job ID.
        """
        page = openai_client.fine_tuning.jobs.list(limit=1)
        return page.data[0].id


    def retrieve_job(self, job_id: str):
        """
            Retrieve a single job record by ID.
        """
        return openai_client.fine_tuning.jobs.retrieve(job_id)


    def list_events(self, job_id: str, limit: int = 10):
        """
            List the last N events for a given fine‑tune job.
        """
        page = openai_client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=limit
        )
        return page.data



    @staticmethod
    def parse_price(text: str):
        """
            Extract the first integer or decimal number from text and return it.
            Returns 0.0 if no number is found.
        """
        clean = text.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", clean)
        return float(match.group()) if match else 0.0


    def build_predictor(self,
                        prompt_fn: Callable[[Any], List[dict]],
                        model_name: str,
                        max_tokens: int = 1,
                        temperature: float = 0.0,
                        seed: int = 42):
        """
            Return a function(item)->float which queries our fine‑tuned `model_name`.
        """
        def predictor(item: Any) -> float:
            messages = prompt_fn(item)
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed
            )
            content = response.choices[0].message.content.strip()
            print("=== RAW RESPONSE ===")
            print(content)
            print("====================")
            return self.parse_price(content)

        #predictor.__name__ = model_name.replace("-", "_")
        return predictor







