import cv2
import numpy as np

CHALLENGE_RAW_OPTIONS_COORDINATES = [
    # 1-3
    [[0, 0], [112, 112]],
    [[116, 0], [228, 112]],
    [[232, 0], [344, 112]],

    # 4-6
    [[0, 116], [112, 228]],
    [[116, 116], [228, 228]],
    [[232, 116], [344, 228]],

    # 7-9
    [[0, 232], [112, 344]],
    [[116, 232], [228, 344]],
    [[232, 232], [344, 344]]
]

CHALLENGE_RAW_HINT_COORDINATES = [[2, 344], [42, 384]]

def crop_v3_nine_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    cropped_images = []
    for coords in CHALLENGE_RAW_OPTIONS_COORDINATES:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        cropped = img[y1:y2, x1:x2]
        cropped_images.append(cropped)

    x1, y1 = CHALLENGE_RAW_HINT_COORDINATES[0]
    x2, y2 = CHALLENGE_RAW_HINT_COORDINATES[1]
    hint_image = img[y1:y2, x1:x2]
    return {
        "options": cropped_images,
        "hint": hint_image
    }


def annote_v3_nine_raw_image(image_bytes, guessed_category_id = -1, default_selection: list[bool] = None):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if default_selection is None:
        selected = [False] * len(CHALLENGE_RAW_OPTIONS_COORDINATES)
    else:
        assert len(default_selection) == len(CHALLENGE_RAW_OPTIONS_COORDINATES)
        selected = default_selection

    input_seq = str(guessed_category_id)

    def with_annotations():
        annoted_img = img.copy()
        for idx, coords in enumerate(CHALLENGE_RAW_OPTIONS_COORDINATES):
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            if selected[idx]:
                cv2.circle(annoted_img, ((x1+x2)//2, (y1+y2)//2), 10, (0, 255, 0), -1)
        
        cv2.putText(annoted_img, input_seq, (10,50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,0), 1)
        return annoted_img

    def imshow():
        cv2.imshow('raw', with_annotations())
        cv2.setMouseCallback('raw', mouse_click)

    def mouse_click(event, x, y, flags, param): 
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, coords in enumerate(CHALLENGE_RAW_OPTIONS_COORDINATES):
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                if x1 <= x and x <= x2 and y1 <= y and y <= y2:
                    selected[idx] = not selected[idx]
            imshow()

    key_pressed = 0
    while key_pressed != ord('n') and key_pressed != ord('q'):
        imshow()
        key_pressed = cv2.waitKey(0)
        if ord('0') <= key_pressed and key_pressed <= ord('9'):
            input_seq += chr(key_pressed)
        elif key_pressed == ord('c'):
            input_seq = ""

    if len(input_seq) == 0:
        input_seq = "-1"

    cv2.destroyWindow('raw')
    return selected, int(input_seq), key_pressed == ord('q')
