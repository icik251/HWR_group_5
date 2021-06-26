import collections
import numpy as np
from utils import load_image, save_image
from character_segmentation.line_processing import LineProcessing
from data_preparation.character_processing import CharacterProcessing
from model import Model
from pathlib import Path
from character_style_recognition.data_prep_tensor import (
    RecognitionDataPrepTensor,
    StyleDataPrepTensor,
)
from line_segmentation import line_segmentation
import os
import argparse

parser = argparse.ArgumentParser(
    description="Run the Hebrew character recognizer and style classifier system"
)
parser.add_argument(
    "images_dir", type=Path, help="Path to the directory where the images that will be processed are located", required=True
)
parser.add_argument(
    "output_dir",
    type=Path,
    help="Path to the directory where the results are going to be saved", required=True
)

char2unicode = {
    "Alef": ("א", "U+05D0"),
    "Ayin": ("ע", "U+05E2"),
    "Bet": ("ב", "U+05D1"),
    "Dalet": ("ד", "U+05D3"),
    "Gimel": ("ג", "U+05D2"),
    "He": ("ה", "U+05D4	"),
    "Het": ("ח", "U+05D7"),
    "Kaf": ("כ", "U+05DB"),
    "Kaf-final": ("ך", "U+05DA"),
    "Lamed": ("ל", "U+05DC"),
    "Mem": ("ם", "U+05DD"),
    "Mem-medial": ("מ", "U+05DE"),
    "Nun-final": ("ן", "U+05DF"),
    "Nun-medial": ("נ", "U+05E0"),
    "Pe": ("פ", "U+05E4"),
    "Pe-final": ("ף", "U+05E3"),
    "Qof": ("ק", "U+05E7"),
    "Resh": ("ר", "U+05E8"),
    "Samekh": ("ס", "U+05E1"),
    "Shin": ("ש", "U+05E9"),
    "Taw": ("ת", "U+05EA"),
    "Tet": ("ט", "U+05D8"),
    "Tsadi-final": ("ץ", "U+05E5"),
    "Tsadi-medial": ("צ", "U+05E6"),
    "Waw": ("ו", "U+05D5"),
    "Yod": ("י", "U+05D9"),
    "Zayin": ("ז", "U+05D6"),
}


def pipeline_logic(images_dir, save_path):

    # Logic for extracing the lines and save them in a structure as follows:
    line_segmentation(images_dir, save_path)
    # data\\images\\image_name_folder
    #    line_1_folder
    #    line_2_folder (and so on)
    #            line_2.png
    #            segmented_char_1.png
    #            segmented_char_2.png
    #            segmented_char_n.png

    # Iterate image folders and their nested folders to get the line

    # Maybe change this to "results"
    result_images_dir = save_path

    # CONSTANT VARIABLES
    # Choose a resizing mode depending on our best model later
    RESIZE_REOCOGNITION = "smallest"
    RESIZE_STYLE = "average"

    # Models paths
    reco_model_path = "config_data\\models\\character_recognition_final\\norm_smallest_batch_32_augmented_train_val_non_augmented_mnist_True_freeze_False_optim_SGD_lr_0.01\\checkpoint_optimal.pth"
    style_model_path = "config_data\\models\\style_classification_final\\norm_avg_batch_300_augmented_train_val_augmented_mnist_True_freeze_False_optim_Adam_lr_0.001\\checkpoint_optimal.pth"

    reco_model_obj = Model(
        mode="recognition",
        model_path_to_load=reco_model_path,
        freeze_layers=False,
        is_production=True,
    )

    style_model_obj = Model(
        mode="style",
        model_path_to_load=style_model_path,
        freeze_layers=False,
        is_production=True,
    )

    for image_dir in os.listdir(result_images_dir):
        # Process each line of the current image and extract characters
        for line_dir in os.listdir(os.path.join(result_images_dir, image_dir)):
            if line_dir.endswith(".txt"):
                continue

            images_in_line_dir_iter = os.listdir(
                os.path.join(result_images_dir, image_dir, line_dir)
            )
            if line_dir.split("_")[0] == "line" and len(images_in_line_dir_iter) > 0:
                for image_in_dir_line in images_in_line_dir_iter:
                    if image_in_dir_line.split("_")[
                        0
                    ] == "line" and image_in_dir_line.endswith(".png"):
                        path_to_line_image = os.path.join(
                            result_images_dir, image_dir, line_dir, image_in_dir_line
                        )
                        line_processing = LineProcessing(path_to_line_image)

                        line_processing.logic()

                # List to save the resized images before they are ran through the network
                list_of_resized_characters_to_recognize = list()
                # List of original images to input to the style classifier resizer
                list_of_original_character_images = list()
                list_of_original_char_images_names = list()

                images_in_line_dir_iter = os.listdir(
                    os.path.join(result_images_dir, image_dir, line_dir)
                )

                for image_in_dir_line in images_in_line_dir_iter:
                    if image_in_dir_line.split("_")[0] == "char":
                        path_to_char_image = os.path.join(
                            result_images_dir, image_dir, line_dir, image_in_dir_line
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
                            resize_mode=RESIZE_REOCOGNITION,
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
                        os.path.join(result_images_dir, image_dir, line_dir),
                    )

    # Save the recognized characters in a file in the image dir
    for image_dir in os.listdir(result_images_dir):
        # Process each line of the current image and extract characters
        line_folders = os.listdir(os.path.join(result_images_dir, image_dir))

        # Sort folders by lines
        temp_dict_of_lines = dict()
        for line_folder_unsorted in line_folders:
            if line_folder_unsorted.endswith(".txt"):
                continue

            temp_dict_of_lines[
                int(line_folder_unsorted.split("_")[1])
            ] = line_folder_unsorted

        temp_dict_of_sorted_lines = collections.OrderedDict(
            sorted(temp_dict_of_lines.items())
        )

        line_folders = list(temp_dict_of_sorted_lines.values())

        for line_dir in line_folders:
            if line_dir.endswith(".txt"):
                continue

            images_in_line_dir_iter = os.listdir(
                os.path.join(result_images_dir, image_dir, line_dir)
            )

            dict_of_characters = dict()
            for curr_file in images_in_line_dir_iter:

                char_name = curr_file.split("_")[0]

                if char_name in char2unicode.keys():

                    end_word = False

                    if len(curr_file.split("_")) >= 4:
                        char_position = int(curr_file.split("_")[2])
                        end_word = True
                    else:
                        char_position = int(curr_file.split("_")[2].split(".")[0])
                        end_word = False

                    dict_of_characters[char_position] = (
                        char2unicode[char_name][0],
                        end_word,
                    )

            ordered_dict_of_chars = collections.OrderedDict(
                sorted(dict_of_characters.items(), reverse=True)
            )

            f_recognized = open(
                os.path.join(
                    result_images_dir, image_dir, "{}_characters.txt".format(image_dir)
                ),
                "a",
                encoding="utf-8",
            )
            builded_string = ""
            for char_pos, (char_symbol, is_end_word) in ordered_dict_of_chars.items():

                # First put the empty space after that the char because we write from right to left
                if is_end_word:
                    builded_string += " "
                    # f_recognized.write(" ")

                builded_string += char_symbol

            f_recognized.write(builded_string)

            f_recognized.write("\n")
            f_recognized.close()

        print(
            "Recognized characters saved in {} for image {}".format(
                os.path.join(
                    result_images_dir, image_dir, "{}_characters.txt".format(image_dir)
                ),
                image_dir,
            )
        )

    # After all lines for all images are recognized, we loop again to classify style
    for image_dir in os.listdir(result_images_dir):

        list_of_image_probabilities = list()

        for line_dir in os.listdir(os.path.join(result_images_dir, image_dir)):
            if line_dir.split("_")[0] == "line" and len(images_in_line_dir_iter) > 0:

                list_of_resized_characters_to_style_classify = list()

                for image_in_dir_line in os.listdir(
                    os.path.join(result_images_dir, image_dir, line_dir)
                ):
                    if (
                        image_in_dir_line.split("_")[0]
                        in style_model_obj.char2idx.keys()
                    ):
                        path_to_recognized_char_image = os.path.join(
                            result_images_dir, image_dir, line_dir, image_in_dir_line
                        )

                        character_processing = CharacterProcessing(
                            path_to_recognized_char_image,
                            resize_mode=RESIZE_STYLE,
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
        linearTransform = (
            lambda probability: (probability - 1 / 3) * (1 - 3 * 0.05) + 1 / 3
        )

        transformed_image_probabilities = list(
            map(lambda x: list(map(linearTransform, x)), list_of_image_probabilities)
        )

        # Naive Bayes Classification of the handwritten page in a single line:
        classified_style_idx = np.argmax(
            sum(list(map(lambda x: np.log(x), transformed_image_probabilities)))
        )
        classified_style_label = list(style_model_obj.style2idx.keys())[
            list(style_model_obj.style2idx.values()).index(classified_style_idx)
        ]

        # Save the style in a txt in the image dir
        with open(
            os.path.join(
                result_images_dir, image_dir, "{}_style.txt".format(image_dir)
            ),
            "w",
        ) as f:
            f.write(classified_style_label)
        f.close()

        print(
            "Classified style saved in {} for image {}".format(
                os.path.join(
                    result_images_dir, image_dir, "{}_style.txt".format(image_dir)
                ),
                image_dir,
            )
        )


def main():
    args = parser.parse_args()
    pipeline_logic(images_dir=Path(args.images_dir), save_path=Path(args.output_dir))


if __name__ == "__main__":
    main()
