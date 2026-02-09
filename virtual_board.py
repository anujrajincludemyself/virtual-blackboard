import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Create a black canvas to draw on
# We'll initialize it with the same size as the camera frame once we read the first frame
canvas = None

# Variables for drawing
xp, yp = 0, 0
drawColor = (255, 0, 255)  # Default color: Pink
brushThickness = 15
eraserThickness = 50

# Color definition
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ERASER = (0, 0, 0)

def fingersUp(lmList):
    """
    Detects which fingers are up based on landmarks.
    :param lmList: List of landmarks (id, x, y)
    :return: List of 5 integers (0 or 1) representing detect fingers (Thumb to Pinky)
    """
    fingers = []
    
    # Thumb (Check if tip is to the right/left of the knuckle depending on hand)
    if lmList[4][0] > lmList[3][0]: 
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers: Index, Middle, Ring, Pinky
    # Tip (8, 12, 16, 20) < PIP Joint (6, 10, 14, 18) (y-coordinates - remember 0 is top)
    tips = [8, 12, 16, 20]
    for tip in tips:
        if lmList[tip][1] < lmList[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers

print("Starting Virtual Board...")
print("Commands:")
print(" - Index Finger Up: Draw")
print(" - Two Fingers Up (Index + Middle): Selection/Hover (Not drawing)")
print(" - Full Hand Open: Eraser (Clears Board)")
print(" - 'q' to Quit")

while True:
    try:
        # 1. Import image
        success, img = cap.read()
        if not success:
            print("Failed to capture image from camera.") 
            break

        # Flip image horizontally for natural interaction
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        # Initialize canvas if not done
        if canvas is None:
            canvas = np.zeros((h, w, 3), np.uint8)

        # 2. Find Hand Landmarks
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        lmList = []
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([cx, cy])

        # 3. Process Landmarks if hand detected
        if len(lmList) != 0:
            x1, y1 = lmList[8]  # Index Finger Tip
            x2, y2 = lmList[12] # Middle Finger Tip
            
            # 4. Check which fingers are up
            fingers = fingersUp(lmList)

            # 5. Selection Mode - Two fingers are up (Index and Middle)
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0 # Reset drawing history
                
                # Check for clicks in header
                if y1 < 125:
                    if 250 < x1 < 450:
                        drawColor = BLUE
                        print("Selected Blue")
                    elif 550 < x1 < 750:
                        drawColor = GREEN
                        print("Selected Green")
                    elif 800 < x1 < 1050:
                        drawColor = RED
                        print("Selected Red")
                    elif 1100 < x1 < 1280:
                        drawColor = ERASER
                        print("Selected Eraser")
                
                cv2.rectangle(img, (x1-25, y1-25), (x2+25, y2+25), drawColor, cv2.FILLED)

            # 6. Drawing Mode - Index finger is up
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                if drawColor == ERASER:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(canvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(canvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                
                xp, yp = x1, y1

            # 7. Eraser Mode (Full Hand - "Duster")
            if all(f == 1 for f in fingers):
                 canvas = np.zeros((h, w, 3), np.uint8)
                 xp, yp = 0, 0
                 print("Canvas Cleared")

        # Draw UI (Color Palette)
        cv2.rectangle(img, (250, 0), (450, 125), BLUE, cv2.FILLED)
        cv2.rectangle(img, (550, 0), (750, 125), GREEN, cv2.FILLED)
        cv2.rectangle(img, (800, 0), (1050, 125), RED, cv2.FILLED)
        cv2.rectangle(img, (1100, 0), (1280, 125), ERASER, cv2.FILLED)
        cv2.rectangle(img, (1100, 0), (1280, 125), (255, 255, 255), 2) # Border
        
        cv2.putText(img, "BLUE", (295, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "GREEN", (585, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "RED", (875, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "ERASER", (1125, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 8. Merge Image and Canvas
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, canvas)

        # Show Image
        cv2.imshow("Virtual Board", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except Exception as e:
        print(f"An error occurred: {e}")
        break

cap.release()
cv2.destroyAllWindows()
