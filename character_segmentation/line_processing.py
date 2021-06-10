import cv2
from utils import convert_image_to_binary, convert_grayscale

class LineProcessing:
    def __init__(self, path_to_image_line) -> None:
        self.initial_line = cv2.imread(path_to_image_line)
        self.gray_image = None
        self.binary_image = None
        self.vert_histogram = None
        self.line_width_array = None
        self.list_of_segmented_characters = list()
        
    def apply_vertical_projection(self, binary_treshold="median"):
        self.gray_image = convert_grayscale(self.initial_line)
        thresh, self.binary_image = convert_image_to_binary(self.gray_image, mode=binary_treshold)
        
        (h, w) = self.binary_image
        self.line_width_array = [0 for z in range(0, w)]

        # Record the peaks of each column
        for j in range(0, w):  # traverse a column
            for i in range(0, h):  # traverse a line
                if self.binary_image[i, j] == 0:  # If the point is changed to a black point
                    self.line_width_array[j] += 1  # The counter in this column is incremented by one
                    self.binary_image[i, j] = 255  # turn it to white after recording
            
        for j in range(0, w):  # Iterate through each column
            for i in range(
                (h - self.line_width_array[j]), h
            ):  # Blacken to the bottom from the top point where the column should turn black
                self.binary_image[i, j] = 0  # black
                
        self.vert_histogram = self.binary_image
        
    def segment_characters(self):
        curr_idx = 0
        self.list_of_segmented_characters = list()
        for column_idx in range(len(self.line_width_array)-2):
            if self.line_width_array[column_idx] < 5 and self.line_width_array[column_idx+1] >=5 and self.line_width_array[column_idx+2] >= 5:
                self.list_of_segmented_characters.append(self.gray_image[:,curr_idx:column_idx])
                curr_idx = column_idx
                
        