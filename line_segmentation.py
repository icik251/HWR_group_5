import cv2
import numpy as np
import os


def line_segmentation(imagepath, debug=False):
    np.set_printoptions(threshold=np.inf)
    path = imagepath
    if not os.path.exists('data/images/'):
        os.makedirs('data/images/')

    for file in os.listdir(path):
        if not file.endswith('.pbm'):
            continue

        savepath = "data/images/" + file[:-4] + "/"

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        image = cv2.imread(path + file)
        im_height = image.shape[0]
        im_width = image.shape[1]

        stripe_width = im_width // 15

        ''' simple segmentation by dilating all text and collecting these as images '''
        # grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(savepath + "gray.png", gray)

        # binary
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        # cv2.imwrite(savepath + 'threshold.png', thresh)

        pieces = gray.copy()
        psl = np.zeros((im_height, 16))
        i = 0
        tiles = []
        '''
        # Divide the document image in stripes of fixed size
        for x in range(0, im_width, stripe_width):
            x1 = x + stripe_width
            tiles = pieces[0:im_height, x:x + stripe_width]

            # If HPP \=2 for any row, then that row is considered as piecewise separating line (PSL)
            for row in range(0, im_height):
                hpp = 0
                for pixel in range(0, tiles.shape[1]):
                    if tiles[row][pixel] == 0:
                        hpp += 1
                    if hpp > 2:
                        psl[row][i] = 1
                        break

            cv2.rectangle(pieces, (x, 0), (x1, im_height), (0, 255, 0))
            cv2.imwrite(savepath + "vertical" + str(x) + ".png", tiles)
            i += 1

        # Consecutive PSLs are reduced to one PSL only
        line_heights = []
        print(psl.shape)
        for x in range(0, 16):
            line_height = 0
            # Average line height (avg_line_height) is computed per strip
            for y in range(0, im_height):
                if psl[y][x] == 1:
                    line_height += 1

                if psl[y][x] == 0 and y != 0 and psl[y-1][x] == 1:
                    line_heights.append(line_height)
                    line_height = 0

        avg_line_height = np.mean(line_heights)
        # Based on avg_line_height, over-segmentation is detected and handled.

        # Based on avg_line_height, under-segmentation is detected and handled.

        # Finally, lines are separated.
        '''

        # dilation
        kernel = np.ones((5, 100), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        # cv2.imwrite(savepath + 'dilation.png', img_dilation)

        # find contours
        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        heights = []
        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
            roi = image[y:y + h, x:x + w]
            heights.append(roi.shape[0])

        avg_line_height = np.mean(heights)
        med_line_height = np.median(heights)

        saved = image.copy()
        deleted = image.copy()

        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
            roi = image[y:y + h, x:x + w]
            # print(roi.shape)
            # print(i)

            # show ROI
            if not os.path.exists(savepath + '/line_%d/' % (i + 1)):
                os.makedirs(savepath + '/line_%d/' % (i + 1))
            if not os.path.exists(savepath + '/line_%d/not_accepted' % (i + 1)):
                os.makedirs(savepath + '/line_%d/not_accepted' % (i + 1))

            if roi.shape[0] < avg_line_height / 2:
                cv2.imwrite(savepath + '/line_%d/not_accepted/line_%d.png' % (i + 1, i + 1), roi)
                cv2.rectangle(deleted, (x, y), (x + w, y + h), (90, 0, 255), 2)
            else:
                cv2.imwrite(savepath + '/line_%d/line_%d.png' % (i + 1, i + 1), roi)
                cv2.rectangle(saved, (x, y), (x + w, y + h), (90, 0, 255), 2)

        # cv2.imwrite(savepath + 'marked_image.png', saved)
        # cv2.imwrite(delpath + 'ignored_image.png', deleted)

        # data = open(savepath + '' + "line_segmentation_debug_data.txt", "w")
        # data.write("avg line height: " + str(avg_line_height))
        # data.write("\n")
        # data.write("median line height: " + str(med_line_height))
        # data.write("\n")
        # data.close()
    return