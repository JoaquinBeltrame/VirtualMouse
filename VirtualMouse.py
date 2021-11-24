import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math

mp_drawing =  mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


capture = cv2.VideoCapture(0)


color_mouse_pointer = (0,255,255)


#Coordenadas de la pantalla:
SCREEN_X_INI = 40
SCREEN_Y_INI = 65
SCREEN_X_FIN = 650
SCREEN_Y_FIN = 900

aspect_ratio_screen = (SCREEN_X_FIN - SCREEN_X_INI)/(SCREEN_Y_FIN - SCREEN_Y_INI)
#print("aspect_ratio_screen:", aspect_ratio_screen)

X_Y_INI = 0


def distance(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1 - p2)


#Detección índice bajo (CLICK IZQ):
def detect_index_down(hand_landmarks):
    index_down = False
    color_base = (255,0,255)
    color_index = (255,255,0)

    #Para acceder a un dedo específico:
    x_base1 = int(hand_landmarks.landmark[0].x*width)
    y_base1 = int(hand_landmarks.landmark[0].y*height)

    x_base2 = int(hand_landmarks.landmark[9].x*width)
    y_base2 = int(hand_landmarks.landmark[9].y*height)

    x_index = int(hand_landmarks.landmark[8].x*width)
    y_index = int(hand_landmarks.landmark[8].y*height)

    dist_base = distance(x_base1, y_base1, x_base2, y_base2)
    dist_base_index = distance(x_base1, y_base1, x_index, y_index)
    
    if dist_base_index < dist_base:
        index_down = True
        color_base = (0,255,0)
        color_index = (0,255,0)

    cv2.circle(output, (x_base1, y_base1), 5, color_base, 2)
    cv2.circle(output, (x_index, y_index), 5, color_index, 2)
    cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(output, (x_base1, y_base1), (x_index, y_index), color_index, 3)

    return index_down


#Detección meñique bajo (CLICK DER):
def detect_pinky_down(hand_landmarks):
    pinky_down = False
    color_base = (255,0,255)
    color_pinky = (255,255,0)

    #Para acceder a un dedo específico:
    x_base1 = int(hand_landmarks.landmark[0].x*width)
    y_base1 = int(hand_landmarks.landmark[0].y*height)

    x_base2 = int(hand_landmarks.landmark[9].x*width)
    y_base2 = int(hand_landmarks.landmark[9].y*height)

    x_pinky = int(hand_landmarks.landmark[20].x*width)
    y_pinky = int(hand_landmarks.landmark[20].y*height)

    dist_base = distance(x_base1, y_base1, x_base2, y_base2)
    dist_base_pinky = distance(x_base1, y_base1, x_pinky, y_pinky)

    if dist_base_pinky < dist_base:
        pinky_down = True
        color_base = (0,255,0)
        color_pinky = (0,255,0)

    cv2.circle(output, (x_base1, y_base1), 5, color_base, 2)
    cv2.circle(output, (x_pinky, y_pinky), 5, color_pinky, 2)
    cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(output, (x_base1, y_base1), (x_pinky, y_pinky), color_pinky, 3)

    return pinky_down


#SCROLL DOWN:
def detect_pair_down(hand_landmarks):
    pair_down = False
    color_base = (255,0,255)
    color_pair_down = (255,255,0)

    #Para acceder a un dedo específico:
    x_base1 = int(hand_landmarks.landmark[0].x*width)
    y_base1 = int(hand_landmarks.landmark[0].y*height)

    x_base2 = int(hand_landmarks.landmark[9].x*width)
    y_base2 = int(hand_landmarks.landmark[9].y*height)

    x_middle = int(hand_landmarks.landmark[12].x*width)
    y_middle = int(hand_landmarks.landmark[12].y*height)

    x_ring = int(hand_landmarks.landmark[16].x*width)
    y_ring = int(hand_landmarks.landmark[16].y*height)

    x_pair, y_pair = (x_middle + x_ring)//2, (y_middle + y_ring)//2

    dist_base = distance(x_base1, y_base1, x_base2, y_base2)
    dist_base_pair = distance(x_base1, y_base1, x_middle, y_middle)
    
    if dist_base_pair < dist_base:
        pair_down = True
        color_base = (0,255,0)
        color_pair_down = (0,255,0)

    cv2.circle(output, (x_base1, y_base1), 5, color_base, 2)
    cv2.circle(output, (x_middle, y_middle), 5, color_pair_down, 2)
    cv2.circle(output, (x_ring, y_ring), 5, color_pair_down, 2)
    cv2.circle(output, (x_pair, y_pair), 5, color_pair_down, 2)
    cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(output, (x_base1, y_base1), (x_pair, y_pair), color_pair_down, 3)
    cv2.line(output, (x_middle, y_middle), (x_ring, y_ring), color_pair_down, 3)

    return pair_down


#SCROLL UP:
def detect_pair_up(hand_landmarks):
    pair_up = False
    color_base = (255,0,255)
    color_pair_up = (255,255,0)
    
    #Para acceder a un dedo específico:
    x_base1 = int(hand_landmarks.landmark[0].x*width)
    y_base1 = int(hand_landmarks.landmark[0].y*height)

    x_base2 = int(hand_landmarks.landmark[9].x*width)
    y_base2 = int(hand_landmarks.landmark[9].y*height)

    x_middle = int(hand_landmarks.landmark[12].x*width)
    y_middle = int(hand_landmarks.landmark[12].y*height)

    x_ring = int(hand_landmarks.landmark[16].x*width)
    y_ring = int(hand_landmarks.landmark[16].y*height)

    x_pair, y_pair = (x_middle + x_ring)//2, (y_middle + y_ring)//2

    length = math.hypot(x_ring - x_middle, y_ring - y_middle)
    #print(length)

    dist_base = distance(x_base1, y_base1, x_base2, y_base2)
    dist_base_pair = distance(x_base1, y_base1, x_middle, y_middle)
    
    if length < 50 and dist_base_pair > dist_base:
        pair_up = True
        color_base = (0,255,0)
        color_pair_up = (0,255,0)

    cv2.circle(output, (x_base1, y_base1), 5, color_base, 2)
    cv2.circle(output, (x_middle, y_middle), 5, color_pair_up, 2)
    cv2.circle(output, (x_ring, y_ring), 5, color_pair_up, 2)
    cv2.circle(output, (x_pair, y_pair), 5, color_pair_up, 2)
    cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(output, (x_base1, y_base1), (x_pair, y_pair), color_pair_up, 3)
    cv2.line(output, (x_middle, y_middle), (x_ring, y_ring), color_pair_up, 3)

    return pair_up


with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.5) as hands:

    #¿Por qué el While? Porque va frame a frame!
    while True:
        ret, frame = capture.read()
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame,1)


        ###Dibujo del área azul:
        area_width = width - X_Y_INI * 2
        area_height = int(area_width / aspect_ratio_screen)
        aux_img = np.zeros(frame.shape, np.uint8)
        aux_img = cv2.rectangle(aux_img, (X_Y_INI, X_Y_INI), (X_Y_INI+area_width, X_Y_INI+area_height), (255,0,0), -1)
        #cv2.imshow("ImagenAuxiliar", aux_img)
        output = cv2.addWeighted(frame, 1, aux_img, 0.7, 0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)


        ###Punto de referencia:
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[9].x*width)
                y = int(hand_landmarks.landmark[9].y*height)

                ###Control de los puntos de referencia:
                x_mouse = np.interp(x, (SCREEN_X_INI, SCREEN_X_FIN), (X_Y_INI, X_Y_INI + area_width))
                y_mouse = np.interp(y, (SCREEN_Y_INI, SCREEN_Y_FIN), (X_Y_INI, X_Y_INI + area_height))


                pyautogui.moveTo(int(x_mouse), int(y_mouse))
                

                ###CLICK IZQUIERDO:
                if detect_index_down(hand_landmarks):
                    pyautogui.click()
                ###CLICK DERECHO:
                if detect_pinky_down(hand_landmarks):
                    pyautogui.click(button="right")
                
                ###SCROLL DOWN MOUSE:
                if detect_pair_down(hand_landmarks):
                    pyautogui.scroll(-1)
                ###SCROLL UP MOUSE:
                elif detect_pair_up(hand_landmarks):
                    pyautogui.scroll(1)


                #cv2.circle(frame, (x,y), 10, color_mouse_pointer, 3)
                cv2.circle(output, (x,y), 10, color_mouse_pointer, 3)
                #cv2.circle(frame, (x,y), 5, color_mouse_pointer, -1)
                cv2.circle(output, (x,y), 5, color_mouse_pointer, -1)

                ###Dibujo de los 21 hand landmarks y sus conexiones:
                #mp_drawing.draw_landmarks(
                #    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                #    mp_drawing.DrawingSpec(color=(0,0,255),thickness=5, circle_radius=5),
                #    mp_drawing.DrawingSpec(color=(0,255,0),thickness=3),
                #)

        #cv2.imshow("CAPTURA", frame)
        #cv2.namedWindow("SALIDA", cv2.WINDOW_NORMAL)
        final_output = cv2.resize(output, (700,660), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("SALIDA", final_output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

capture.release()
cv2.destroyAllWindows()