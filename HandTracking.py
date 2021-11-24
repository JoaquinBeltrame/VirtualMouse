import cv2
import mediapipe as mp

#Para dibujar las Hand.Landmarks y sus conexiones):
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


#Captura por cámara:
capture = cv2.VideoCapture(0)


#Configuración inicial:
with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 1,
    min_detection_confidence = 0.5) as hands:

    while True:
        ret, frame = capture.read()
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame,1)

        #Ajuste de color para detección:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        #Siempre que haya una mano...
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                #print(hand_landmarks)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=3))


        cv2.imshow("VideoCaptura", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

capture.release()
cv2.destroyAllWindows()