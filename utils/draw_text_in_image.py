import cv2

"""
 Draws text in image
 img : cv2.imread
 text : "Image: " + ground_truth_img[0] + " "
 pos : (margin, v_pos)
 color : (255,255,255)
 line_width : int
"""


def draw_text_in_image(img, text: str, pos: tuple, color: tuple, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCornerOfText = pos
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)

if __name__ == '__main__':
    img = cv2.imread("/Users/jonghwanpark/PycharmProjects/mAP/mAP/input/images-optional/2007_000027.jpg")
    text = "Image: " + '2007_000027.jpg' + " "
    pos = (10,100)
    color = (255,255,255)
    line_width = 0

    print(draw_text_in_image(img,text,pos,color,line_width))