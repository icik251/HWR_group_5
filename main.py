from src.character_segmentation import line_segmentation as ls
# from src.character_segmentation import char_segmentation

# from src.character_recognition import sift

# from src.style_classification import classifier

debug = False

binarized_path = "../binarized/"

ls.line_segmentation("binarized_path", debug)
