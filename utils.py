import cv2 
from textblob import TextBlob

def preprocess_image(image):
    denoised_image = cv2.fastNlMeansDenoisingColored( 
        image, None, 15, 10, 9, 21
    )
	
    return denoised_image


def postprocess_text(word_token):
    """
    params:
    - word_tokenize
    methods:
    - remove short word {word > 1}
    - remove symbol
    - ignore numerical word
    output:
    - concatenated text
    """
    text = []
    for word in word_token:
        if len(word) > 1 and word.isalpha():
            if word.isnumeric():
                text.append(word)
            elif word.isalnum():
                text.append(word)
            else:
                pass
        elif word.isnumeric():
            text.append(word)
        else:
            pass

    result = " ".join(text)
    result = str(TextBlob(result).correct()) 
	
    return result


def getAngle(image):
    grayImg  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImg  = cv2.GaussianBlur(grayImg, (9, 9), 0)
    bwImg 	 = cv2.threshold(blurImg, 210, 230, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    cleanImg = cv2.fastNlMeansDenoising(bwImg, None, 15, 7, 21 )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 4))
    dilate = cv2.dilate(cleanImg, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
	
    return -1.0 * angle


def rotateImage(image, angle):
    (h,w)  = image.shape[:2]
    center = (w//2, h//2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
	
    rotatedImg = cv2.warpAffine(image, matrix, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	
    return rotatedImg


def deskew(image):
    angle = getAngle(image)
	
    return rotateImage(image, -1.0 * angle)

