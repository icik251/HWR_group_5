import cv2
import os
from data_loader import DataLoader
import numpy as np
from skimage.morphology import medial_axis, skeletonize
import matplotlib.pyplot as plt
import copy


class CharacterProcessing:
    def __init__(self, dict_of_chars) -> None:
        self.dict_of_chars = dict_of_chars
        self.dict_of_structuring_elements = dict()

    @staticmethod
    def crop_white_spaces_image(image):
        # Load image, grayscale, Gaussian blur, Otsu's threshold
        original = image.copy()
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (25, 25), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Perform morph operations, first open to remove noise, then close to combine
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)

        # Find enclosing boundingbox and crop ROI
        coords = cv2.findNonZero(close)
        x, y, w, h = cv2.boundingRect(coords)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        crop = original[y : y + h, x : x + w]

        return crop
    
    @staticmethod
    def crop_white_spaces_image_v2(image):
        gray = image
        gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
        coords = cv2.findNonZero(gray) # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        rect = image[y:y+h, x:x+w] # Crop the image - note we do this on the original image
        
        return rect

    def resize_image(self, image, shape: tuple):
        return cv2.resize(image, shape)

    def crop_white_spaces(self, list_of_samples):
        list_of_result = list()
        for sample in list_of_samples:
            list_of_result.append(CharacterProcessing.crop_white_spaces_image(sample))
        return list_of_result

    def normalize_character_images_size(self, letter, list_of_samples, save_mode=True):
        new_shape = [0, 0]
        for sample in list_of_samples:
            # resize method expects (width, height), that's why
            # we reverse them here already
            new_shape[0] += sample.shape[1]
            new_shape[1] += sample.shape[0]

        new_shape = np.divide(new_shape, len(list_of_samples))
        new_shape = list(np.array(new_shape).astype("uint8"))
        new_shape = tuple(new_shape)

        list_of_normalized_images = list()

        for idx, sample in enumerate(list_of_samples):
            resized_img = self.resize_image(sample, new_shape)
            # convert to binary (not working here because there are no intersections,
            # in the case of 300 examples like Alef)
            threshold, resized_img_binary = CharacterProcessing.convert_image_to_binary(
                resized_img, mode="median"
            )

            if save_mode:
                self.save_normalized_character_images(letter, idx, resized_img_binary)
            list_of_normalized_images.append(resized_img_binary)

        return list_of_normalized_images

    def save_normalized_character_images(self, letter, img_name, img):
        if not os.path.exists(os.path.join("data\\resized_binary", letter)):
            os.makedirs(os.path.join("data\\resized_binary", letter))
        cv2.imwrite(
            os.path.join("data\\resized_binary", letter, "{}.pgm".format(img_name)), img
        )

    @staticmethod
    def convert_image_to_binary(image, mode="median"):
        if mode == "median":
            median_value = np.median(image)
            threshold = int(median_value / 1.33)

            (_, image_binary) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        elif isinstance(mode, int):
            (_, image_binary) = cv2.threshold(image, mode, 255, cv2.THRESH_BINARY)
            threshold = mode

        return int(threshold), image_binary

    def create_alpha_structuring_elements(self, save_mode=False, threshold_mode=5):
        for letter, list_of_samples in self.dict_of_chars.items():

            list_of_samples_cropped = self.crop_white_spaces(list_of_samples)

            list_of_samples_resized = self.normalize_character_images_size(
                letter, list_of_samples_cropped, save_mode=save_mode
            )

            for idx, train_sample in enumerate(list_of_samples_resized):
                # init structuring element from first sample
                if idx == 0:
                    structuring_element = list_of_samples_resized[0].astype("uint32")
                    continue

                for i in range(train_sample.shape[0]):
                    try:
                        structuring_element[i, :] = np.add(
                            structuring_element[i, :], train_sample[i, :]
                        )
                    except Exception as e:
                        print(e)

            structuring_element = np.divide(
                structuring_element, len(list_of_samples_resized)
            )
            structuring_element = np.rint(structuring_element).astype("uint8")

            threshold, structuring_element_binary = CharacterProcessing.convert_image_to_binary(
                structuring_element, mode=threshold_mode
            )

            self.dict_of_structuring_elements[letter] = (
                threshold,
                structuring_element_binary,
            )

            self.save_alpha_structuring_element(
                letter, threshold, structuring_element_binary
            )

    def save_alpha_structuring_element(
        self, letter, threshold, structuring_element_binary
    ):
        if not os.path.exists(os.path.join("data\\alpha_structuring_elements_binary")):
            os.makedirs(os.path.join("data\\alpha_structuring_elements_binary"))

        cv2.imwrite(
            os.path.join(
                "data\\alpha_structuring_elements_binary",
                "{}_{}.pgm".format(letter, threshold),
            ),
            structuring_element_binary,
        )

    def create_final_structuring_element(self):
        for letter, (
            threshold,
            structuring_element,
        ) in self.dict_of_structuring_elements.items():
            # reverse black and white (important for skeleton with medial axis)
            structuring_element[structuring_element == 0] = 1
            structuring_element[structuring_element == 255] = 0
            skel, distance = medial_axis(structuring_element, return_distance=True)
            # skel = skeletonize(structuring_element, method='lee')

            # kernel = np.ones((5, 5), np.uint8)
            # skel = cv2.erode(structuring_element, kernel, iterations=1)

            skel = skel.astype("uint8")
            skel[skel == 0] = 255
            skel[skel == 1] = 0

            # skel_rgb = cv2.cvtColor(skel, cv2.COLOR_BGR2RGB)

            self.save_final_structuring_element(letter, threshold, skel)

            """
            dist_on_skel = distance * skel

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.imshow(structuring_element, cmap=plt.cm.gray, interpolation="nearest")
            ax1.axis("off")
            ax2.imshow(dist_on_skel, cmap=plt.cm.plasma, interpolation="nearest")
            ax2.contour(structuring_element, [0.5], colors="w")
            ax2.axis("off")

            fig.subplots_adjust(
                hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1
            )
            plt.show()
            """

    def save_final_structuring_element(self, letter, threshold, structuring_element):
        if not os.path.exists(os.path.join("data\\final_structuring_elements_binary")):
            os.makedirs(os.path.join("data\\final_structuring_elements_binary"))

        cv2.imwrite(
            os.path.join(
                "data\\final_structuring_elements_binary",
                "{}_{}.pgm".format(letter, threshold),
            ),
            structuring_element,
        )

    def get_structuring_elements(self):
        return self.dict_of_structuring_elements

    def build_structuring_elements(self, save_normalized=False, threshold_mode=5):
        self.create_alpha_structuring_elements(save_normalized, threshold_mode)
        self.create_final_structuring_element()


data_loader = DataLoader()
dict_result = data_loader.get_characters_train_data(
    path="D:\\PythonProjects\\HWR_group_5\\data\\character_set_labeled"
)

for i in range(1, 11):
    char_processing = CharacterProcessing(copy.deepcopy(dict_result))
    char_processing.build_structuring_elements(save_normalized=True, threshold_mode=i)
