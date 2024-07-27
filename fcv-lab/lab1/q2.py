import cv2

# Write a simple program to read and display a video file

vid = cv2.VideoCapture('q2_video.mp4')

while True:
    ret, frame = vid.read()
    if ret:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
