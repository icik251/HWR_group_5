import numpy as np
from utils import load_image, save_image
from character_segmentation.line_processing import LineProcessing
from data_preparation.character_processing import CharacterProcessing
from model import Model
from character_style_recognition.data_prep_tensor import (
    RecognitionDataPrepTensor,
    StyleDataPrepTensor,
)
import os


if __name__ == "__main__":

    path_to_real_scrolls = "D:\\PythonProjects\\HWR_group_5\\data\\sample-test-2021\\"

    # Logic for extracing the lines and save them in a structure as follows:

    # image_name_folder
    #    line_1_folder
    #    line_2_folder (and so on)
    #            line_2.png
    #            segmented_char_1.png
    #            segmented_char_2.png
    #            segmented_char_n.png

    # Iterate image folders and their nested folders to get the lines
    mock_images_dir = "data\\mock_real_images"
    # Choose a resizing mode depending on our best model later
    RESIZE_MODE = "smallest"

    reco_model_obj = Model(
        mode="recognition",
        model_path_to_load="data\\models\\character_recognition\\norm_smallest_freeze_False\\checkpoint_optimal.pth",
        freeze_layers=False,
        is_production=True,
    )

    style_model_obj = Model(
        mode="style",
        model_path_to_load="data\\models\\style_classification\\norm_smallest_freeze_False\\checkpoint_optimal.pth",
        freeze_layers=False,
        is_production=True,
    )

    for image_dir in os.listdir(mock_images_dir):
        # Process each line of the current image and extract characters
        for line_dir in os.listdir(os.path.join(mock_images_dir, image_dir)):
            if line_dir.split("_")[0] == "line":
                for image_in_dir_line in os.listdir(
                    os.path.join(mock_images_dir, image_dir, line_dir)
                ):
                    if image_in_dir_line.split("_")[
                        0
                    ] == "line" and image_in_dir_line.endswith(".png"):
                        path_to_line_image = os.path.join(
                            mock_images_dir, image_dir, line_dir, image_in_dir_line
                        )
                        line_processing = LineProcessing(path_to_line_image)

                        line_processing.logic()
                        print(
                            "Characters extracted from image {} from line: {}".format(
                                image_dir, image_in_dir_line
                            )
                        )
                # List to save the resized images before they are ran through the network
                list_of_resized_characters_to_recognize = list()
                # List of original images to input to the style classifier resizer
                list_of_original_character_images = list()
                list_of_original_char_images_names = list()

                for image_in_dir_line in os.listdir(
                    os.path.join(mock_images_dir, image_dir, line_dir)
                ):
                    if image_in_dir_line.split("_")[0] == "char":
                        path_to_char_image = os.path.join(
                            mock_images_dir, image_dir, line_dir, image_in_dir_line
                        )

                        # Append original image
                        list_of_original_character_images.append(
                            load_image(path_to_char_image)
                        )
                        list_of_original_char_images_names.append(
                            image_in_dir_line.split(".")[0]
                        )

                        character_processing = CharacterProcessing(
                            path_to_char_image,
                            resize_mode=RESIZE_MODE,
                            model_mode="recognition",
                        )
                        # Append resized for recognition
                        character_processing.resize_image()
                        list_of_resized_characters_to_recognize.append(
                            character_processing.get_image()
                        )

                reco_data_prep_tensor = RecognitionDataPrepTensor()
                reco_production_dataloader = (
                    reco_data_prep_tensor.get_data_loader_production(
                        list_of_resized_characters_to_recognize
                    )
                )

                list_of_reco_predictions = reco_model_obj.recognize_character(
                    reco_production_dataloader
                )

                # Save predicted images with label in the line folder
                for original_char_image, original_char_img_name, pred_label in zip(
                    list_of_original_character_images,
                    list_of_original_char_images_names,
                    list_of_reco_predictions,
                ):
                    save_image(
                        original_char_image,
                        "{}_{}".format(pred_label, original_char_img_name),
                        os.path.join(mock_images_dir, image_dir, line_dir),
                    )
                    
    # After all lines for all images are recognized, we loop again to classify style
    for image_dir in os.listdir(mock_images_dir):
        
        list_of_image_probabilities = list()
        
        for line_dir in os.listdir(os.path.join(mock_images_dir, image_dir)):
            if line_dir.split("_")[0] == "line":
                
                list_of_resized_characters_to_style_classify = list()
                
                for image_in_dir_line in os.listdir(
                    os.path.join(mock_images_dir, image_dir, line_dir)
                ):      
                    if image_in_dir_line.split("_")[0] in style_model_obj.char2idx.keys():
                        path_to_recognized_char_image = os.path.join(
                            mock_images_dir, image_dir, line_dir, image_in_dir_line
                        )
                        
                        character_processing = CharacterProcessing(
                            path_to_recognized_char_image,
                            resize_mode=RESIZE_MODE,
                            model_mode="style",
                        )
                        
                        # Append resized for style classification
                        character_processing.resize_image()
                        list_of_resized_characters_to_style_classify.append(
                            character_processing.get_image()
                        )
                        
                style_data_prep_tensor = StyleDataPrepTensor()
                style_production_dataloader = (
                    style_data_prep_tensor.get_data_loader_production(
                        list_of_resized_characters_to_style_classify
                    )
                )

                list_of_style_probability_for_line = style_model_obj.classify_style(
                    style_production_dataloader
                )
                
                list_of_image_probabilities += list_of_style_probability_for_line
        
        # Naive Bayes and get style name
        print(list_of_image_probabilities)
        classified_style_idx = np.argmax(sum(list(map(lambda x: np.log(x), list_of_image_probabilities))))
        classified_style_label = list(style_model_obj.style2idx.keys())[
                    list(style_model_obj.style2idx.values()).index(classified_style_idx)
                ]
        
        # Save the style in a txt in the image dir
        with open(os.path.join(mock_images_dir, image_dir, "classified_style.txt"), "w") as f:
            f.write(classified_style_label)
        f.close()
                        
                
        
                    
                
