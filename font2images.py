# Uses pillow (you can also use another imaging library if you want)
from PIL import Image, ImageFont, ImageDraw

# Character mapping for each of the 27 tokens
char_map = {
    "Alef": ")",
    "Ayin": "(",
    "Bet": "b",
    "Dalet": "d",
    "Gimel": "g",
    "He": "x",
    "Het": "h",
    "Kaf": "k",
    "Kaf-final": "\\",
    "Lamed": "l",
    "Mem": "{",
    "Mem-medial": "m",
    "Nun-final": "}",
    "Nun-medial": "n",
    "Pe": "p",
    "Pe-final": "v",
    "Qof": "q",
    "Resh": "r",
    "Samekh": "s",
    "Shin": "$",
    "Taw": "t",
    "Tet": "+",
    "Tsadi-final": "j",
    "Tsadi-medial": "c",
    "Waw": "w",
    "Yod": "y",
    "Zayin": "z",
}

# Returns a grayscale image based on specified label of img_size
def create_image(label, img_size, font_size):
    if label not in char_map:
        raise KeyError("Unknown label!")

    # Create blank image and create a draw interface
    img = Image.new("L", img_size, 255)
    draw = ImageDraw.Draw(img)

    # Get size of the font and draw the token in the center of the blank image
    w, h = font_size.getsize(char_map[label])
    draw.text(
        ((img_size[0] - w) / 2, (img_size[1] - h) / 2), char_map[label], 0, font_size
    )

    return img


import os

# Create a 50x50 image of the Alef token and save it to disk
# To get the raw data cast it to a numpy array

list_of_chras = char_map.keys()

for char in list_of_chras:
    font_size = ImageFont.truetype("data\\Habbakuk.ttf", 50)

    img = create_image(char, (50 + 8, 50 + 8), font_size)

    #if not os.path.exists("data\\generated_characters\\{}".format(char)):
    #    os.makedirs("data\\generated_characters\\{}".format(char))
    img.save(
        "data\\generated_characters\\{}_font_{}.png".format(char, 50+8)
    )
