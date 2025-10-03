import yaml
import subprocess
import time
import logging
import sys

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration for the Grid Search ---
CONFIG_PATH = 'config.yaml'
MAIN_SCRIPT_PATH = 'main.py'
PRESET_TO_TUNE = 'deepface_vgg'  # The preset we want to test

# The list of eps values you want to automatically test
EPS_VALUES_TO_TEST = [0.4, 0.45, 0.5, 0.525, 0.55, 0.575, 0.6, 0.65]


def run_grid_search():
    """
    Automates testing a list of eps values for a given preset.
    """
    logging.info(f"--- Starting Grid Search for '{PRESET_TO_TUNE}' ---")

    try:
        with open(CONFIG_PATH, 'r') as f:
            original_config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Could not read original config file. Aborting. Error: {e}")
        return

    for eps_value in EPS_VALUES_TO_TEST:
        logging.info(f"\n{'=' * 50}")
        logging.info(f"--- Testing {PRESET_TO_TUNE} with eps = {eps_value} ---")
        logging.info(f"{'=' * 50}\n")

        try:
            config = original_config.copy()  # Start with a fresh copy each time
            config['active_model'] = PRESET_TO_TUNE
            config['presets'][PRESET_TO_TUNE]['clustering']['eps'] = eps_value

            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, sort_keys=False)

            logging.info(f"Updated config.yaml for eps={eps_value}")

            subprocess.run([sys.executable, MAIN_SCRIPT_PATH], check=True)

        except Exception as e:
            logging.error(f"Run failed for eps={eps_value}. Error: {e}")

        time.sleep(2)

    # Restore the original config file when done
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(original_config, f, sort_keys=False)
    logging.info("\n--- Grid Search complete. Original config.yaml restored. ---")


if __name__ == "__main__":
    run_grid_search()