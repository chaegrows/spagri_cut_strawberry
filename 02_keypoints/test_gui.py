# test_highgui.py
import cv2, numpy as np
img = np.zeros((200, 300, 3), np.uint8)
win = "TEST"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)   # WINDOW_GUI_NORMAL 쓰지 마세요(4.5.5에선 불안정)
cv2.imshow(win, img)
cv2.waitKey(500)                          # 창이 뜰 시간을 줌 (중요)
print("visible?", cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE))
cv2.setMouseCallback(win, lambda *a: None)  # 콜백 시도
cv2.waitKey(0)
cv2.destroyAllWindows()
