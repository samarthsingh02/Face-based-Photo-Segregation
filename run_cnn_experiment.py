import yaml
import subprocess
import time
import logging
import sys  # <-- 1. ADD THIS IMPORT

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for the Experiment ---
CONFIG_PATH = 'config.yaml'
MAIN_SCRIPT_PATH = 'main.py'
RESIZE_WIDTHS_TO_TEST = [1000, 800, 600, 500, 450, 400, 350, 300, 250, 200]  # Add or change the sizes you want to test


def run_experiment():
    """
    Automates the process of running the main script with different resize_width
    settings for the CNN model.
    """
    logging.info("--- Starting CNN Resize Width Experiment ---")

    # Read the original config file to restore it later
    try:
        with open(CONFIG_PATH, 'r') as f:
            original_config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Could not read original config file. Aborting. Error: {e}")
        return

    for width in RESIZE_WIDTHS_TO_TEST:
        logging.info(f"\n{'=' * 50}")
        logging.info(f"--- Running experiment for resize_width: {width}px ---")
        logging.info(f"{'=' * 50}\n")

        try:
            # Modify the config for the current run
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)

            config['active_model'] = 'cnn_accurate'
            config['presets']['cnn_accurate']['resize_width'] = width

            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, sort_keys=False)

            logging.info(f"Updated config.yaml for resize_width: {width}px")

            # --- 2. MAKE THIS CHANGE ---
            # Run the main script using the correct Python interpreter from the venv
            subprocess.run([sys.executable, MAIN_SCRIPT_PATH], check=True)

        except subprocess.CalledProcessError as e:
            logging.error(f"The main script failed for resize_width: {width}px. Error: {e}")
            logging.error("Continuing to the next experiment...")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

        time.sleep(5)  # A small pause between runs

    # Restore the original config file when all experiments are done
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(original_config, f, sort_keys=False)
    logging.info("\n--- All experiments complete. Original config.yaml has been restored. ---")


if __name__ == "__main__":
    run_experiment()