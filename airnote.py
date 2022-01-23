import cv2
import numpy as np
import time

from cvzone.HandTrackingModule import HandDetector
from random import randint

capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

canvas = np.zeros((720, 1280, 3), np.uint8)

x1n, x2n, y1n, y2n = 1280, 1, 1080, 101
x1e, x2e, y1e, y2e = 200, 1, 1, 101
x1l, x2l, y1l, y2l = 800, 1, 600, 101
letters_list = ['a', 'A', 'b', 'B', 'c', 'C', 'd', 'D', 'e', 'E', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'j', 'J', 'k',
                'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U',
                'w', 'W', 'x', 'X', 'y', 'Y', 'z', 'Z']

prev_x, prev_y = 0, 0

change = False


def draw_another_letter(letters_list):
    if letters_list:
        random_letter_index = randint(0, len(letters_list) - 1)
        letter = letters_list.pop(random_letter_index)
    else:
        letter = 'WELL DONE'
    yield letter


letter = draw_another_letter(letters_list).__next__()

while True:
    mode = 'selection'
    success, frame = capture.read()
    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame, flipType=False)
    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        index_finger = lmList1[8]
        middle_finger = lmList1[12]
        length, info, frame = detector.findDistance(index_finger, middle_finger, frame)
        if length < 50:
            mode = 'selection'
            prev_x, prev_y = 0, 0
        elif length >= 50:
            mode = 'painting'
        if mode == 'painting':
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = index_finger[0], index_finger[1]
            cv2.line(frame, (prev_x, prev_y), (index_finger[0], index_finger[1]), (255, 255, 255), 12)
            cv2.line(canvas, (prev_x, prev_y), (index_finger[0], index_finger[1]), (255, 255, 255), 12)
            prev_x, prev_y = index_finger[0], index_finger[1]
            change = False
        elif mode == 'selection':
            if y1e < index_finger[0] < x1e and x2e < index_finger[1] < y2e:
                time.sleep(.3)
                cv2.destroyAllWindows()
                quit()
            elif y1n < index_finger[0] < x1n and x2n < index_finger[1] < y2n and change is False:
                change = True
                letter = draw_another_letter(letters_list).__next__()
                canvas = np.zeros((720, 1280, 3), np.uint8)
                mode = 'painting'

    exit_frame = cv2.rectangle(frame, (x1e, x2e), (y1e, y2e), (0, 0, 0), cv2.BORDER_WRAP)
    cv2.putText(exit_frame, "EXIT", (x1e - 130, y2e - 50), cv2.FONT_HERSHEY_PLAIN, 2, (36, 255, 12), 2)
    next_frame = cv2.rectangle(frame, (x1n, x2n), (y1n, y2n), (0, 0, 0), cv2.BORDER_WRAP)
    cv2.putText(next_frame, "NEXT", (x1n - 130, y2n - 50), cv2.FONT_HERSHEY_PLAIN, 2, (36, 255, 12), 2)
    letter_frame = cv2.rectangle(frame, (x1l, x2l), (y1l, y2l), (0, 0, 0), cv2.BORDER_WRAP)
    cv2.putText(letter_frame, letter, (x1l - 100, y2l - 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    gray_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inverted = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    img_inverted = cv2.cvtColor(img_inverted, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inverted)
    frame = cv2.bitwise_or(frame, canvas)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
