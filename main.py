import cv2
import mediapipe as mp
from math import sqrt
import pygame
from settings import *
from time import sleep

cap = cv2.VideoCapture(1)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
pygame.init()
display=1

clickMax=35

display_info = pygame.display.get_desktop_sizes()
if len(display_info) < 2:
    print("No second screen connected. Using the default one.")
    display=0
screen_width, screen_height = display_info[display]
screen = pygame.display.set_mode((screen_width, screen_height), display=display)
running=True

def locatePoint(frame, area_min=10, area_max=500):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    punti = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area_min < area < area_max:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                punti.append((cx, cy))
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    return punti



screen.fill((0, 0, 0))
pygame.draw.circle(screen, (255, 255, 255), startPos, 10)
pygame.draw.circle(screen, (255, 255, 255), (endPos[0],startPos[1]), 10)
pygame.draw.circle(screen, (255, 255, 255), endPos, 10)
pygame.draw.circle(screen, (255, 255, 255), (startPos[0],endPos[1]), 10)
pygame.display.flip()
sleep(0.3)
_, frame = cap.read()
points=locatePoint(frame)
startCam=(min(p[0] for p in points), min(p[1] for p in points))
endCam=(max(p[0] for p in points), max(p[1] for p in points))
cv2.rectangle(frame, startCam, endCam, (0, 0, 255), 2)
cv2.imshow("Rilevazione Puntini", frame)
cv2.waitKey(1)
sleep(2)
cv2.destroyWindow("Rilevazione Puntini")


while running:
    for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    screen.fill((0, 0, 0))

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
        
            tp=(0, 0) # Thumb position
            ip=(0, 0) # Index position
            d=0       # Thumb-Index distance
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                if id==4:
                    tp=(lm.x*w, lm.y*h)
                if id==8:
                    ip=(lm.x*w, lm.y*h)
                    
            cp=(max(0, min(int((ip[0]+tp[0])/2)-startCam[0], endCam[0]-startCam[0])), max(0, min(int((ip[1]+tp[1])/2)-startCam[1], endCam[1]-startCam[1])))
            d=sqrt((tp[0]-ip[0])**2 + (tp[1]-ip[1])**2)
            print(cp, tp[1], ip[1], end="\t")
            r=6
            if d<clickMax:
                r=12
            cv2.circle(img, (cp[0]+startCam[0], cp[1]+startCam[1]), r, (255, 255, 0), cv2.FILLED)
            pos=(startPos[0]+(cp[0]/(endCam[0]-startCam[0]))*(endPos[0]-startPos[0]), startPos[1]+(cp[1]/(endCam[1]-startCam[1]))*(endPos[1]-startPos[1]))
            print(cp, pos, end="\t")
            pygame.draw.circle(screen, (0, 255, 255), pos, r)
    print(end="                          \r")
    pygame.display.flip()
    #cv2.imshow("Image", img)
    #cv2.waitKey(1)

pygame.quit()
print()