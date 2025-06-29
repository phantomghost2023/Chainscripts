# This is a simulated large processing script

import time
import random


def process_large_dataset(data_size):
    dataset = [random.random() for _ in range(data_size)]
    processed_data = []
    for item in dataset:
        # Simulate complex processing
        time.sleep(0.00001)  # Small delay to simulate work
        processed_data.append(item * 2 + 1)
    return processed_data


if __name__ == "__main__":
    print("Starting large data processing...")
    result = process_large_dataset(50000)  # Process 50,000 items
    print(f"Finished processing {len(result)} items.")
