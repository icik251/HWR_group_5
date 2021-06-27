# Handwriting Recognition of Hebrew Group 5

# Prerequisites
1. Install Python 3.8.5.

2. Install PyTorch from: https://pytorch.org/get-started/locally/ depending on your system and environment. Make sure to install a CPU version. 

3. Use the following line in the command prompt to install the rest of the dependencies:
pip install -r requirements.txt

4. Unzip the folder "config_data" in the main project folder. The name and the structure of the folder should not be changed. Download link: https://drive.google.com/file/d/1dizI-9QNatSiQR17w8Fl30cTacJE6ZgB/view

# How to run the recognizer and style classifier system

To run the program, run

python main.py [input_folder] [output_folder] where\
[input_folder] is the name of the folder containing the images and \
[output_folder] is the name of the folder you want the output to be found\
For example: python main.py "input_images" "output_images"

main.py is the file, which runs the whole pipeline. Iteratively each image will be processed in the following way:
1. Line segmentation
2. Character and word segmentation
3. Recognition of segmented characters.
4. Style classification for each recognized character in an image, followed by applying Naive Bayes on the probabilities for all of the characters, which results in the most probable style of the image

## Code Structure

The root level consists of the following file: 
- `main.py` which launches the entire pipeline

In the `line_segmentation.py` file one can find the code for the line segmentation step.
In the `character_segmentation` folder one can find the files required for the character segmentation step.
In the `character_style_recognition` folder one can find the files required for the character recognition and style classification step.
In the `data_preparation` folder one can find the files required for the data preparation step.

## Authors

We are Kenneth Muller, Gabriel Leuenberger, Hristo Stanulov and Kaan Yesildal. This is a project as part of the Handwriting Recognition course.
