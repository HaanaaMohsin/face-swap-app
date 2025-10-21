import os
import sys
import subprocess
import argparse

# Define the path to the Streamlit app
script_path = os.path.join(os.path.dirname(__file__), 'app.py')

def run_streamlit():
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])


def main():
    parser = argparse.ArgumentParser(description="Face Swap App runner")
    parser.add_argument("mode", nargs="?", default="ui", choices=["ui"], help="Run mode")
    args = parser.parse_args()

    if args.mode == "ui":
        run_streamlit()


if __name__ == "__main__":
    main()
