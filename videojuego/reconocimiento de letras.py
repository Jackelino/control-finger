# Visión artificial
# Profesora Vianney
# Semestre 2022A 06 de Junio del 2022
# Ricardo Gonzalez Garduño
# Irvin Carapia Hernández
# José
# Importación de las librerías
import cv2
import mediapipe as mp
import numpy as np
from math import *
import pygame
import constants
import levels
from player import Player

# Importación de las herramientas de media pipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Solicitar acceso a la camara de video
cap = cv2.VideoCapture(0)

# Definir las dimensiones y propiedades de la cámara
wCam, hCam = 800, 600
cap.set(3,wCam)
cap.set(4,hCam)

up = False
down = False
count = 0

# Iniciamos pygame
pygame.init()

# Valores de la pantalla en el videojuego
size = [constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Juego de plataforma con visión artificial")

# Se crea al jugador
player = Player()

# Se crean los dos niveles
level_list = []
level_list.append(levels.Level_01(player))
level_list.append(levels.Level_02(player))

# Se indica que se comienza en el nivel uno
current_level_no = 0
current_level = level_list[current_level_no]

active_sprite_list = pygame.sprite.Group()
player.level = current_level

player.rect.x = 340
player.rect.y = constants.SCREEN_HEIGHT - player.rect.height
active_sprite_list.add(player)

# Se inicia el loop hasta salir de la camara con ESC
done = False

# Con esto se actualizara el videojuego
clock = pygame.time.Clock()

# Inicia el loop del videojuego y reconocimiento de imagen
with mp_hands.Hands(
    # Se indica que se trabajará con una imagen en movimiento y solo una mano
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75) as hands:

    while True:
            ret, frame = cap.read()
            if ret == False:
                break
            height, width, _ = frame.shape
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks is not None:    
            # Accediendo a los puntos de referencia, de acuerdo a su nombre
                for hand_landmarks in results.multi_hand_landmarks:

                    # COORDENADAS MEÑIQUE
                    x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
                    y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)
                    x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * width)
                    y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * height)
                    x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * width)
                    y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * height)
                    
                    #COORDENADAS ANULAR
                    x4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
                    y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)
                    x5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * width)
                    y5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * height)
                    x6 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * width)
                    y6 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * height)

                    #COORDENADAS MEDIO
                    x7 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
                    y7 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
                    x8 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * width)
                    y8 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * height)
                    x9 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * width)
                    y9 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * height)

                    #COORDENADAS INDICE
                    x10 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                    y10 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                    x11 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * width)
                    y11 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * height)
                    x12 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width)
                    y12 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height)

                    #COORDENADAS PULGAR PARTE EXTERNA
                    x13 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                    y13 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
                    x14 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * width)
                    y14 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * height)
                    x15 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * width)
                    y15 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * height)

                    #COORDENADAS PULGAR PARTE INTERNA
                    x16 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                    y16 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
                    x17 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * width)
                    y17 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * height)
                    x18 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                    y18 = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)

                    p1 = np.array([x1, y1])
                    p2 = np.array([x2, y2])
                    p3 = np.array([x3, y3])

                    p4 = np.array([x4, y4])
                    p5 = np.array([x5, y5])
                    p6 = np.array([x6, y6])

                    p7 = np.array([x7, y7])
                    p8 = np.array([x8, y8])
                    p9 = np.array([x9, y9])

                    p10 = np.array([x10, y10])
                    p11 = np.array([x11, y11])
                    p12 = np.array([x12, y12])

                    p13 = np.array([x13, y13])
                    p14 = np.array([x14, y14])
                    p15 = np.array([x15, y15])

                    p16 = np.array([x16, y16])
                    p17 = np.array([x17, y17])
                    p18 = np.array([x18, y18])

                    l1 = np.linalg.norm(p2 - p3)
                    l2 = np.linalg.norm(p1 - p3)
                    l3 = np.linalg.norm(p1 - p2)

                    l4 = np.linalg.norm(p5 - p6)
                    l5 = np.linalg.norm(p4 - p6)
                    l6 = np.linalg.norm(p4 - p5)

                    l7 = np.linalg.norm(p8 - p9)
                    l8 = np.linalg.norm(p7 - p9)
                    l9 = np.linalg.norm(p7 - p8)

                    l10 = np.linalg.norm(p11 - p12)
                    l11 = np.linalg.norm(p10 - p12)
                    l12 = np.linalg.norm(p10 - p11)

                    l13 = np.linalg.norm(p14 - p15)
                    l14 = np.linalg.norm(p13 - p15)
                    l15 = np.linalg.norm(p13 - p14)

                    l16 = np.linalg.norm(p17 - p18)
                    l17 = np.linalg.norm(p16 - p18)
                    l18 = np.linalg.norm(p16 - p17)


                    #despejar y quitar determinaciones con numeradores menores al denominador
                    
                    num_den1= (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)

                    num_den2= (l4**2 + l6**2 - l5**2) / (2 * l4 * l6)

                    num_den3= (l7**2 + l9**2 - l8**2) / (2 * l7 * l9)

                    num_den4= (l10**2 + l12**2 - l11**2) / (2 * l10 * l12)

                    num_den5= (l13**2 + l15**2 - l14**2) / (2 * l13 * l15)

                    num_den6 = (l16**2 + l18**2 - l17**2) / (2 * l16 * l18)



                    # Calcular el ángulo

                    if l1 and l3 != 0 and -1<num_den1<1:
                        angle1 = round(degrees(abs(acos(num_den1))))
                    else:
                        angle1 = 0

                    if l4 and l6 != 0 and -1<num_den2<1:
                        angle2 = round(degrees(abs(acos(num_den2))))
                    else:
                        angle2 = 0

                    if l7 and l9 != 0 and -1<num_den3<1:
                        angle3 = round(degrees(abs(acos(num_den3))))
                    else: 
                        angle3 = 0

                    if l10 and l12 != 0 and -1<num_den4<1:
                        angle4 = round(degrees(abs(acos(num_den4))))
                    else:
                        angle4 = 0

                    if l13 and l15 != 0 and -1<num_den5<1: 
                        angle5  = round(degrees(abs(acos(num_den5))))
                    else:
                        angle5 = 0
                    
                    if l16 and l18 != 0 and -1<num_den6<1:
                        angle6 = round(degrees(abs(acos(num_den6))))
                    else:
                        angle6 = 0

                    

                    angulosid = [angle1, angle2, angle3, angle4, angle5, angle6]
                    dedos = []
                    
                    #pulgar externo angle
                    if angle6 > 125:
                        dedos.append(1)
                    else:
                        dedos.append(0)

                    #pulgar interno
                    if angle5 > 150:
                        dedos.append(1)
                    else:
                        dedos.append(0)

                    #4 dedos
                    for id in range(0,4):
                         if angulosid[id] > 90:
                             dedos.append(1)
                         else:
                             dedos.append(0)

                    TotalDedos = dedos.count(1)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if dedos == [1,1,0,0,0,0]:
                        cv2.rectangle(frame,(0,0),(100,100),(255,255,255), -1)
                        cv2.putText(frame,'A',(20,80),font,3,(0,0,0),2,cv2.LINE_AA)
                        player.go_left()
                    if dedos == [0,0,1,0,0,0]:
                        cv2.rectangle(frame,(0,0),(100,100),(255,255,255), -1)
                        cv2.putText(frame,'I',(20,80),font,3,(0,0,0),2,cv2.LINE_AA)
                        player.go_right()
                    if dedos ==[1,1,0,0,0,1]:
                        cv2.rectangle(frame,(0,0),(100,100),(255,255,255), -1)
                        cv2.putText(frame,'L',(20,80),font,3,(0,0,0),2,cv2.LINE_AA)
                        player.jump()
                    if dedos ==[1,1,1,0,0,0]:
                        cv2.rectangle(frame,(0,0),(100,100),(255,255,255), -1)
                        cv2.putText(frame,'Y',(20,80),font,3,(0,0,0),2,cv2.LINE_AA)
                        player.go_right()
                    if dedos==[0, 1, 0, 0, 1, 1]:
                        cv2.rectangle(frame,(0,0),(100,100),(255,255,255), -1)
                        cv2.putText(frame,'V',(20,80),font,3,(0,0,0),2,cv2.LINE_AA)
                        player.stop()

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        active_sprite_list.update()

                        # Update items in the level
                        current_level.update()

                        # If the player gets near the right side, shift the world left (-x)
                        if player.rect.x >= 500:
                            diff = player.rect.x - 500
                            player.rect.x = 500
                            current_level.shift_world(-diff)

                        # If the player gets near the left side, shift the world right (+x)
                        if player.rect.x <= 120:
                            diff = 120 - player.rect.x
                            player.rect.x = 120
                            current_level.shift_world(diff)

                        # If the player gets to the end of the level, go to the next level
                        current_position = player.rect.x + current_level.world_shift
                        if current_position < current_level.level_limit:
                            player.rect.x = 120
                            if current_level_no < len(level_list) - 1:
                                current_level_no += 1
                                current_level = level_list[current_level_no]
                                player.level = current_level

                        # ALL CODE TO DRAW SHOULD GO BELOW THIS COMMENT
                        current_level.draw(screen)
                        active_sprite_list.draw(screen)

                        # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

                        # Limit to 60 frames per second
                        clock.tick(60)

                        # Go ahead and update the screen with what we've drawn.
                        pygame.display.flip()

            cv2.imshow('Frame',frame)



            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Be IDLE friendly. If you forget this line, the program will 'hang'
            # on exit.

    pygame.quit()



cap.release()
cv2.destroyAllWindows()



