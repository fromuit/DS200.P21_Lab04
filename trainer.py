import os
import torch
import torch.nn as nn
import torch.optim as optim
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import SimpleCNN
from stream import load_stream_data_into_rdd, STREAM_INPUT_DIR

# Ensure directory exists for model saving
MODEL_SAVE_DIR = "./models/saved"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train_partition(iterator, model_state_dict, learning_rate, epochs_per_partition):
    """
    Trains a model locally on a single Spark partition's data.
    Runs on Spark worker nodes.

    Args:
        iterator: An iterator over (image_tensor, label) tuples for this partition.
        model_state_dict: The state dictionary of the global model to start training from.
        learning_rate: Learning rate for the optimizer.
        epochs_per_partition: Number of local epochs to train on this partition's data.

    Returns:
        The state dictionary of the model after training on this partition.
    """
    # Load the model and its state dictionary
    model = SimpleCNN()
    model.load_state_dict(model_state_dict)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare data for local training (convert iterator to list and then to tensors)
    # Note: Collecting partition data to a list might consume significant memory
    # for large partitions. In production, consider more advanced techniques
    # or micro-batching within the partition.
    partition_data = list(iterator)
    if not partition_data:
        print("Partition is empty, returning initial model state.")
        return model.state_dict()

    # Convert list of tuples to separate lists
    images, labels = zip(*partition_data)

    # Stack tensors to create batches for local training
    # The incoming tensors are already [C, H, W]
    images_tensor = torch.stack(images) # Shape [N, C, H, W]
    labels_tensor = torch.tensor(labels, dtype=torch.long) # Shape [N]

    # Local training loop on the partition's data
    model.train()
    for epoch in range(epochs_per_partition):
        optimizer.zero_grad()
        outputs = model(images_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
        # print(f"  Partition trained for epoch {epoch+1}/{epochs_per_partition}, Loss: {loss.item():.4f}") # Optional: log per partition epoch

    print(f"Finished training on partition ({len(partition_data)} samples).")

    # Return the state dictionary of the trained model from this partition
    return model.state_dict()

def average_model_state_dicts(state_dict_list):
    """
    Averages the weights from multiple model state dictionaries.
    This is a simple aggregation strategy.
    """
    if not state_dict_list:
        return None

    # Start with the first state dict
    avg_state_dict = state_dict_list[0].copy()

    # Sum up the weights from the remaining state dicts
    for state_dict in state_dict_list[1:]:
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key]

    # Divide by the number of state dicts to get the average
    num_models = len(state_dict_list)
    for key in avg_state_dict:
        avg_state_dict[key] /= num_models

    print(f"Averaged weights from {num_models} partition models.")
    return avg_state_dict

def evaluate_model(model: nn.Module, data_rdd, spark_context: SparkContext):
    """
    Evaluates the model on a Spark RDD.
    Collects data to the driver for evaluation (simplified approach).
    For large evaluation sets, evaluate per partition.
    """
    print("Collecting data for evaluation...")
    try:
        # This can be memory intensive for large RDDs
        all_data = data_rdd.collect()
        if not all_data:
             print("No data to evaluate on.")
             return 0.0
        images, labels = zip(*all_data)
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        print(f"Collected {len(all_data)} samples for evaluation.")
    except Exception as e:
        print(f"Error collecting data for evaluation: {e}")
        return 0.0


    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    # Using a batch size for evaluation on the driver CPU
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(images_tensor), eval_batch_size):
            batch_images = images_tensor[i:i+eval_batch_size]
            batch_labels = labels_tensor[i:i+eval_batch_size]
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    print(f'Evaluation Accuracy: {100 * accuracy:.2f}%')
    return accuracy

def run_spark_training(spark_context: SparkContext, spark_session: SparkSession,
                       stream_dir=STREAM_INPUT_DIR,
                       epochs=5,
                       learning_rate=0.0005,
                       epochs_per_partition=1,
                       num_partitions=None):
    """
    Main function to orchestrate Spark-based CNN training.
    """
    print("Starting Spark training...")

    # Initialize a model on the driver to get the initial state dict
    global_model = SimpleCNN()
    initial_state_dict = global_model.state_dict()

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # Simulate receiving/loading a new chunk of data (or re-read all available data)
        # In a real stream, you'd load just the new data since the last iteration.
        # Here, for simplicity, we reload all data files present in the directory.
        data_rdd = load_stream_data_into_rdd(spark_context, stream_dir)

        if data_rdd.isEmpty():
            print("No data received in this epoch, skipping training step.")
            continue

        # Repartition the RDD for training distribution
        if num_partitions is None:
             # Use number of available cores/executors by default
             num_partitions = spark_context.defaultParallelism
             print(f"Using default number of partitions: {num_partitions}")

        data_rdd = data_rdd.repartition(num_partitions)
        print(f"RDD repartioned to {data_rdd.getNumPartitions()} partitions.")

        # Get the current state dict to broadcast to workers
        current_state_dict = global_model.state_dict()
        # Broadcast the model state dictionary to all workers
        state_dict_b = spark_context.broadcast(current_state_dict)


        # --- Distributed Training Step ---
        # Train models on each partition using mapPartitions
        # Each worker gets a copy of the state_dict
        # Each worker trains on its partition's data and returns the trained state dict
        partition_trained_state_dicts_rdd = data_rdd.mapPartitions(
            lambda iterator: [train_partition(
                iterator,
                state_dict_b.value, # Access the broadcasted state dict
                learning_rate,
                epochs_per_partition
            )] # mapPartitions expects an iterator to be returned
        )

        # Collect the trained state dictionaries from all partitions to the driver
        print("Collecting trained model state dictionaries from partitions...")
        try:
            partition_trained_state_dicts = partition_trained_state_dicts_rdd.collect()
            print(f"Collected {len(partition_trained_state_dicts)} state dictionaries.")
        except Exception as e:
            print(f"Error collecting partition models: {e}")
            # Handle errors, maybe retry or skip epoch
            continue


        # --- Aggregation Step ---
        # Average the model weights collected from the workers
        if partition_trained_state_dicts:
            avg_state_dict = average_model_state_dicts(partition_trained_state_dicts)
            # Update the global model with the averaged weights
            global_model.load_state_dict(avg_state_dict)
            print("Global model updated with averaged weights.")
        else:
            print("No partition models collected, global model not updated.")

        # --- Evaluation Step (Simplified) ---
        # Evaluate the current global model on the data received in this epoch
        # Note: For robust evaluation, you'd typically use a separate test set,
        # potentially also loaded into Spark.
        print(f"Evaluating model after epoch {epoch+1}...")
        evaluate_model(global_model, data_rdd, spark_context)


    print("\nSpark training finished.")

    # Save the final trained model
    model_path = os.path.join(MODEL_SAVE_DIR, "final_cnn_model.pth")
    torch.save(global_model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")


if __name__ == '__main__':
     # This block is mainly for testing the trainer logic directly,
     # main.py will be the entry point for the full simulation.
     print("Running trainer.py directly for testing...")

     # Setup Spark context for testing
     conf = SparkConf().setAppName("TrainerTest").setMaster("local[*]")
     sc = SparkContext(conf=conf)
     spark = SparkSession(sc)

     # Assuming stream data exists for testing purposes
     # In the full run, stream.py would run first.
     print("Note: Assuming stream data is already generated in ./data/streaming_input")
     print("Please run stream.py or the sender simulation in main.py first.")

     # Run training with some sample parameters
     run_spark_training(sc, spark, epochs=2, epochs_per_partition=1, num_partitions=4)

     sc.stop()
     spark.stop()