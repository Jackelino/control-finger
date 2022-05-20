import cv2
import mediapipe as mp
import os

nameDirectory = 'signal_left'
path = '../resources/entrenamiento'
directory = path + '/' + nameDirectory

cont = 0
numberCapture = 300
positions = []

if not os.path.exists(directory):
    print("carpeta creada", directory)
    os.makedirs(directory)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        copia = frame.copy()
        if ret == False:
            break
        height, width, _ = frame.shape
        # frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                for (id, lm) in enumerate(hand_landmarks.landmark):
                    cordx, cordy = int(lm.x * width), int(
                        lm.y * height)  # obtenemos los puntos de posiscion de cada dedo
                    positions.append([id, cordx, cordy])
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if len(positions) != 0:
                    pto_1 = positions[4]
                    pto_2 = positions[20]
                    pto_3 = positions[12]
                    pto_4 = positions[0]
                    pto_5 = positions[9]  # púnto central
                    x1, y1 = (pto_5[1] - 100), (pto_5[2] - 100)
                    # recortamos la imagen aparttir del punto central para que se observe solo la mano
                    width, height = (x1 + 100), (y1 + 100)
                    # dibujamos un rectango para delimitar la mano
                    x2, y2 = x1 + width, y1 + height
                    dedosReg = copia[y1:y2, x1:x2]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                dedosReg = cv2.resize(dedosReg, (255, 200), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(directory + "/signal_{}.jpg".format(cont), dedosReg)
                cont += 1

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27 or cont == numberCapture:
            break
cap.release()
cv2.destroyAllWindows()
