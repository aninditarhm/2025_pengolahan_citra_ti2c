import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 5
click_distance_threshold = 40
drag_threshold = 0.5
click_hold_threshold = 0.6
########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Status klik & drag
click_left_state = False
click_right_state = False
last_click_time = 0
click_count = 0

drag_active = False
drag_start_time = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    # Mengambil frame & deteksi tangan
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        # Koordinat ujung jari telunjuk & tengah
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        fingers = detector.fingersUp()

        # Mode gerak kursor (1 jari: telunjuk saja)
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

            # Reset drag jika sedang aktif
            if drag_active:
                autopy.mouse.toggle(down=False)
                print("Drag stopped")
                drag_active = False
                drag_start_time = 0

        # Klik kiri & Drag (2 jari: telunjuk + tengah)
        elif fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            current_time = time.time()

            if length < click_distance_threshold:
                # Deteksi drag
                if not drag_active:
                    if drag_start_time == 0:
                        drag_start_time = current_time
                    elif current_time - drag_start_time > drag_threshold:
                        autopy.mouse.toggle(down=True)
                        drag_active = True
                        print("Drag started")
                else:
                    # Menggerakkan mouse saat drag aktif
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    autopy.mouse.move(wScr - clocX, clocY)
                    plocX, plocY = clocX, clocY

                # Klik (bukan drag)
                if not click_left_state and not drag_active:
                    if current_time - last_click_time < click_hold_threshold:
                        click_count += 1
                    else:
                        click_count = 1
                    last_click_time = current_time
                    click_left_state = True

                    if click_count == 2:
                        autopy.mouse.click()
                        print("Double click")
                    else:
                        autopy.mouse.click()
                        print("Single click")
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            else:
                # Reset jika jari menjauh
                click_left_state = False
                drag_start_time = 0
                if drag_active:
                    autopy.mouse.toggle(down=False)
                    print("Drag stopped")
                    drag_active = False

        # Klik kanan (3 jari: jari telunjuk, jari tengah, jari manis)
        if fingers == [0, 1, 1, 1, 0]:
            if not click_right_state:
                autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
                print("Right click")
                click_right_state = True
        else:
            click_right_state = False

    # Gambar frame
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    # Menampilkan FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # Menampilkan ke layar
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()