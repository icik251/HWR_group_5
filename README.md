# Handwriting Recognition of Hebrew Group 5

# Prerequisites
1. Install Python 3.8.5.

2. Install PyTorch from: https://pytorch.org/get-started/locally/ depending on your system and environment.

3. Use the following line in the command prompt to install the rest of the dependencies:
pip install -r requirements.txt

4. Unzip the folder "config_data" in the main project folder
Download link: "https://drive.google.com/file/d/1dizI-9QNatSiQR17w8Fl30cTacJE6ZgB/view

# How to run the recognizer and style classifier system

main.py is the file, which runs the whole pipeline. Iteratively each image will be processed in the following way:
1. Line semgnation
2. Character and word segmantation
3. Recognition of segmented characters.
4. Style classification for each recognized character in an image, followed by applying Naive Bayes on the probabilities for all of the characters, which results in the most probable style of the image

The required arguments to run main.py. These arguments are positional, therefore you need to pass them in the following order: 
1. images_dir - Path to the directory where the images that will be processed are located.
2. output_dir - Path to the directory where the results are going to be saved.
