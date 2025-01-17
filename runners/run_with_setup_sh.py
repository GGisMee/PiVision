import subprocess

def run_shell_script(script_path):
    """Run a shell script."""
    try:
        subprocess.run(['bash', script_path], check=True)
        print(f"Successfully ran {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        exit(1)

def run_python_script(script_path, flags):
    """Run the Python script with the given flags."""
    try:
        # Run the detection script with the hardcoded flags
        subprocess.run(['python','-m', script_path] + flags.split(), check=True)
        print(f"Successfully ran {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        exit(1)

if __name__ == "__main__":
    # Hardcoded flags you want to pass to the detection.py script
    detection_flags = ['', '--input rpi'] # "--input rpi --network yolov6n"
    detection_flag = detection_flags[1]
    # Set the paths to the scripts
    setup_script = './setup_env.sh'  # Adjust the path if needed
    detection_script = 'basic_pipelines.detection'  # Adjust the path if needed

    # Run the setup script
    # run_shell_script(setup_script)

    # Run the detection script with the hardcoded flags
    run_python_script(detection_script, detection_flag)
