import subprocess
import os

import subprocess
import os

def open_image_in_vscode(image_path):
    # Ensure the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file '{image_path}' does not exist.")

    # Open the image in VSCode
    try:
        subprocess.run(["code", image_path], check=True)
        print(f"Opened '{image_path}' in VSCode.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to open '{image_path}' in VSCode. Error: {e}")
    except FileNotFoundError:
        print("The 'code' command is not found. Make sure VSCode is installed and added to PATH.")



if __name__ == "__main__":
    # Change this path to your image file
    image_path = "resources/testing_img.png"
    open_image_in_vscode(image_path)
