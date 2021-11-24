import cv2
import numpy as np
import pyautogui

#Otra vez, el while es porque va frame a frame: 
while True:
    screenshot = pyautogui.screenshot(region=(40, 65, 650, 900))
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    cv2.imshow("SCREENSHOT", screenshot)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()