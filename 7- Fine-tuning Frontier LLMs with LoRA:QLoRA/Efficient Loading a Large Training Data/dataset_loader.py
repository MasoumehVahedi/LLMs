from tqdm import tqdm
from typing import List
from datetime import datetime
from datasets import load_dataset
from product_price_prediction import ProductExample
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


CHUNK_SIZE = 1000
MIN_PRICE = 0.5
MAX_PRICE = 999.50


class DatasetLoader:
    """
        A parallelized, chunked loader for very large HuggingFace Datasets.

        Responsibilities:
          1. fetch a full split via `load_dataset()`
          2. iterate over it in batches of `CHUNK_SIZE`
          3. map each raw row → a TrainingExample (or any subclass we supply)
          4. drop invalid / filtered examples
          5. return one flat list of valid examples

        This lets us avoid OOMs on gigantic datasets, and speeds up preprocessing
        by farm-out to multiple worker processes.
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = None


    def fromDatasetExample(self, dataset_example):
        """
            Wrap one row in our example class.
            If the constructor marks it invalid (e.g. missing fields, out of length bounds),
            we get back None.
        """

        try:
            price_str = dataset_example["price"]
            if price_str:
                price = float(price_str)
                if MIN_PRICE <= price <= MAX_PRICE:
                    record = ProductExample(dataset_example, price)
                    return record if record.valid else None
        except ValueError:
            return None

    def fromChunk(self, chunk):
        """
            Turn one slice of the HF dataset into a list of valid examples.
        """
        batch = []
        for dataset_example in chunk:
            result = self.fromDatasetExample(dataset_example)
            if result:
                batch.append(result)
        return batch


    def chunkGenerator(self):
        """
            Iterate over the Dataset, yielding chunks of datapoints at a time
        """
        total_size = len(self.dataset)
        for i in range(0, total_size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, total_size)))


    def loadInParallel(self, num_workers: int):
        """
           Use concurrent.futures to farm out the work to process chunks of dataset_example.
           This speeds up processing significantly
        """
        results = []
        chunk_count = (len(self.dataset_name) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for batch in tqdm(pool.map(self.fromChunk, self.chunkGenerator()), total=chunk_count):
                results.extend(batch)

        for result in results:
            result.category = self.dataset_name
        return results



    def load(self, num_workers: int = 8) -> List:
        """
           1. Download the HF dataset split.
           2. Map it → our example class in parallel.
           3. Return a flattened list of valid examples.

           - num_workers: is parameter specifies how many processes
             should work on loading and scrubbing the data
        """

        start = datetime.now()
        print(f"Loading dataset {self.dataset_name}", flush=True)
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{self.dataset_name}", split="full", trust_remote_code=True)
        results = self.loadInParallel(num_workers)
        end = datetime.now()
        print(f"Completed {self.dataset_name} with {len(results):,} dataset_example in {(end - start).total_seconds() / 60:.1f} mins", flush=True)
        return results





