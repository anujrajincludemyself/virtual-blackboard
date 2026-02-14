import cv2
import numpy as np
import mediapipe as mp
import os
import math
from shapes_3d import Cube, Sphere, Pyramid, Cylinder

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

# 3D Shapes variables
shape_mode = False
current_shape_type = "cube"  # cube, sphere, pyramid, cylinder
current_shape = None
placed_shapes = []  # List of (shape_object, color) tuples
shape_size = 100
shape_rotation_speed = 0.05

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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

def create_shape(shape_type, position, size):
    """Create a 3D shape object based on type"""
    if shape_type == "cube":
        return Cube(position, size)
    elif shape_type == "sphere":
        return Sphere(position, size)
    elif shape_type == "pyramid":
        return Pyramid(position, size)
    elif shape_type == "cylinder":
        return Cylinder(position, size)
    return None

print("Starting Virtual Board...")
print("Commands:")
print(" - Index Finger Up: Draw")
print(" - Two Fingers Up (Index + Middle): Selection/Hover (Not drawing)")
print(" - Three Fingers Up (Index + Middle + Ring): Shape Mode")
print(" - Pinch (Thumb + Index): Resize shape in Shape Mode")
print(" - Hand Tilt: Rotate shape in Shape Mode")
print(" - Fist (All fingers down): Place shape on canvas")
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
            x3, y3 = lmList[16] # Ring Finger Tip
            x_thumb, y_thumb = lmList[4]  # Thumb Tip
            
            # 4. Check which fingers are up
            fingers = fingersUp(lmList)

            # 5. Three Fingers Mode - Shape Mode
            if fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
                shape_mode = True
                xp, yp = 0, 0  # Reset drawing history
                
                # Check for shape selection in bottom toolbar
                if y1 > 595:
                    if 50 < x1 < 250:
                        current_shape_type = "cube"
                        print("Selected Cube")
                    elif 300 < x1 < 500:
                        current_shape_type = "sphere"
                        print("Selected Sphere")
                    elif 550 < x1 < 750:
                        current_shape_type = "pyramid"
                        print("Selected Pyramid")
                    elif 800 < x1 < 1000:
                        current_shape_type = "cylinder"
                        print("Selected Cylinder")
                
                # Create or update current shape
                if current_shape is None:
                    current_shape = create_shape(current_shape_type, (x1, y1, 0), shape_size)
                else:
                    current_shape.position[0] = x1
                    current_shape.position[1] = y1
                
                # Pinch gesture for size control
                pinch_distance = calculate_distance((x_thumb, y_thumb), (x1, y1))
                shape_size = max(50, min(300, pinch_distance * 2))
                current_shape.size = shape_size
                
                # Hand tilt for rotation (using wrist to middle finger angle)
                wrist_x, wrist_y = lmList[0]
                angle_x = (wrist_y - y2) / 100.0
                angle_y = (wrist_x - x2) / 100.0
                current_shape.rotation[0] += angle_x * shape_rotation_speed
                current_shape.rotation[1] += angle_y * shape_rotation_speed
                
                # Visual feedback
                cv2.circle(img, (x1, y1), 20, (0, 255, 255), cv2.FILLED)
                cv2.putText(img, f"Size: {int(shape_size)}", (x1 + 30, y1), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 6. Fist gesture - Place shape
            elif not any(fingers):
                if shape_mode and current_shape is not None:
                    # Place the shape on canvas
                    placed_shapes.append((current_shape, drawColor))
                    print(f"Placed {current_shape_type}")
                    current_shape = None
                    shape_mode = False

            # 7. Selection Mode - Two fingers are up (Index and Middle)
            elif fingers[1] and fingers[2] and not fingers[3]:
                shape_mode = False
                current_shape = None
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

            # 8. Drawing Mode - Index finger is up (only if not in shape mode)
            if fingers[1] and not fingers[2] and not shape_mode:
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

            # 9. Eraser Mode (Full Hand - "Duster")
            if all(f == 1 for f in fingers):
                 canvas = np.zeros((h, w, 3), np.uint8)
                 placed_shapes = []  # Clear all placed shapes
                 current_shape = None
                 shape_mode = False
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

        # Draw Shape Toolbar (Bottom)
        cv2.rectangle(img, (50, 595), (250, 720), (100, 100, 100), cv2.FILLED)
        cv2.rectangle(img, (300, 595), (500, 720), (100, 100, 100), cv2.FILLED)
        cv2.rectangle(img, (550, 595), (750, 720), (100, 100, 100), cv2.FILLED)
        cv2.rectangle(img, (800, 595), (1000, 720), (100, 100, 100), cv2.FILLED)
        
        # Highlight selected shape
        if current_shape_type == "cube":
            cv2.rectangle(img, (50, 595), (250, 720), (0, 255, 255), 5)
        elif current_shape_type == "sphere":
            cv2.rectangle(img, (300, 595), (500, 720), (0, 255, 255), 5)
        elif current_shape_type == "pyramid":
            cv2.rectangle(img, (550, 595), (750, 720), (0, 255, 255), 5)
        elif current_shape_type == "cylinder":
            cv2.rectangle(img, (800, 595), (1000, 720), (0, 255, 255), 5)
        
        cv2.putText(img, "CUBE", (110, 665), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "SPHERE", (340, 665), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "PYRAMID", (575, 665), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "CYLINDER", (820, 665), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Render placed shapes on canvas
        for shape, color in placed_shapes:
            shape.render(canvas, color, 2)
        
        # Render current shape being manipulated
        if current_shape is not None and shape_mode:
            current_shape.render(img, (0, 255, 255), 3)

        # 10. Merge Image and Canvas
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
