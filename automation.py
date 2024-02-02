import cv2
import numpy as np
from tensorflow.keras.models import load_model
from time import sleep
from PIL import ImageGrab
import pyautogui


def nothing(x):
    pass

def determine_shot(ball_x_center, middle_x, x1, x2, count, handedness):
    if ball_x_center < middle_x and ball_x_center > x1 + (middle_x - x1) * 0.70:
        return  ["s", "down"]
    if ball_x_center < middle_x and ball_x_center > x1 + (middle_x - x1) * 0.50:
        if handedness == "Left":
            return  ["s", "down", "left"]
        else:
            return  ["s", "down"]
    if ball_x_center < middle_x:
        return  ["s", "right"]
    elif ball_x_center > middle_x and ball_x_center < middle_x + (x2 - middle_x) * 0.70:
        return  ["s", "down"]
    elif ball_x_center > middle_x and ball_x_center < middle_x + (x2 - middle_x) * 0.50:
        if handedness == "Right":
            return  ["s", "down", "right"]
        else:
            return  ["s", "down"]
    elif ball_x_center > middle_x:
        return  ["s", "left"]

def determine_shot_old(ball_x_center, middle_x):
    if ball_x_center < middle_x:
        return "Left Shot"
    else:
        return "Right Shot"
model = load_model('model.h5', compile=False)

cv_window_name = "Ball Tracking and Classification"
cv2.namedWindow(cv_window_name)
cv2.createTrackbar('x1', cv_window_name, 166, 1000, nothing)
cv2.createTrackbar('y1', cv_window_name, 175, 1000, nothing)
cv2.createTrackbar('x2', cv_window_name, 489, 1000, nothing)
cv2.createTrackbar('y2', cv_window_name, 420, 1000, nothing)


screen_size = pyautogui.size()
scale_factor = 0.5
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('tracked_video1.avi', fourcc, 20.0, (screen_size.width, screen_size.height))

template1 = cv2.imread('w1.png')
template1_gray = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template1_w, template1_h = template1_gray.shape[::-1]

template2 = cv2.imread('w2.png')
template2_gray = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
template2_w, template2_h = template2_gray.shape[::-1]

template3 = cv2.imread('w3.png')
template3_gray = cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)
template3_w, template3_h = template3_gray.shape[::-1]


ball_positions = []
count=0
while True:
    screenshot = ImageGrab.grab()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    x1 = cv2.getTrackbarPos('x1', cv_window_name)
    y1 = cv2.getTrackbarPos('y1', cv_window_name)
    x2 = cv2.getTrackbarPos('x2', cv_window_name)
    y2 = cv2.getTrackbarPos('y2', cv_window_name)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    middle_x = (x1 + x2) // 2
    cv2.line(frame, (middle_x, y1), (middle_x, y2), (255, 0, 0), 2)
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    prediction = model.predict(np.expand_dims(normalized_frame, axis=0))
    predicted_class = np.argmax(prediction)

    class_names = ['Left', 'Not Active', 'Right']
    class_name = class_names[predicted_class]

    cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if class_name in ['Left', 'Right']:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect_area = gray_frame[y1:y2, x1:x2]
        res = cv2.matchTemplate(rect_area, template1_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        template_w = template1_w
        template_h = template1_h
        if max_val < 0.75:
            res = cv2.matchTemplate(rect_area, template2_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            template_w = template2_w
            template_h = template2_h
        if max_val < 0.75:
            res = cv2.matchTemplate(rect_area, template3_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            template_w = template3_w
            template_h = template3_h
        if max_val > 0.75:
            top_left = (max_loc[0] + x1, max_loc[1] + y1)
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
            aoi_width = x2 - x1
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)


            ball_x_center = (top_left[0] + bottom_right[0]) // 2
            ball_y_center = (top_left[1] + bottom_right[1]) // 2
            if (ball_y_center - y1) < (y2 - y1) * 0.4:
                count +=1
                shot_to_play = determine_shot(ball_x_center, middle_x, x1, x2, count, class_name)
                print(shot_to_play)
                if len(shot_to_play) == 3:
                    pyautogui.keyDown('shift')
                    pyautogui.hotkey(shot_to_play[0],shot_to_play[1],shot_to_play[2])
                elif len(shot_to_play) == 2:
                        pyautogui.keyDown('shift')
                        pyautogui.keyDown(shot_to_play[1])
                        pyautogui.press('s')
                        pyautogui.keyUp('shift')
                        pyautogui.keyUp(shot_to_play[1])
                else:
                    pyautogui.keyDown('shift')
                    pyautogui.press('s')
                cv2.putText(frame, str(shot_to_play), (frame.shape[1]//2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if ball_x_center < middle_x:
                ball_position = 'Left'
            else:
                ball_position = 'Right'

            cv2.putText(frame, ball_position, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ball_positions.append((ball_x_center, top_left[1] + template_h // 2))
        else:
            ball_positions = []

    for i in range(1, len(ball_positions)):
        cv2.line(frame, ball_positions[i - 1], ball_positions[i], (255, 0, 0, 128), 2)
    
    out.write(frame)

    frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)))
    cv2.imshow(cv_window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
