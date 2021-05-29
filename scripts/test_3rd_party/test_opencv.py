import cv2

cap = cv2.VideoCapture(0)
i = 0

while True:
  ret, frame = cap.read()
  key = cv2.waitKey(1)
  if key == 27: break
  cv2.imshow('cap', frame)

cap.release()
cv2.destroyAllWindows()