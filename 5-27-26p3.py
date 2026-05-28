import cv2

video_source = "cars.mp4"
cascade_path = "cars.xml"

cap = cv2.VideoCapture(video_source)

car_cascade = cv2.CascadeClassifier(cascade_path)

if car_cascade.empty():
    print("Error loading cascade classifier")
    exit()

while True:
    ret, frames = cap.read()

    if not ret:
        print("End of video or cannot read from source")
        break
    
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray,scaleFactor= 2,minNeighbors= 0)

    for (x,y,w,h) in cars:
        car_region = frames[y:y+h, x:x+w]
        gray_car = cv2.cvtColor(car_region, cv2.COLOR_BGR2GRAY)
        gray_roi_3ch = cv2.cvtColor(gray_car, cv2.COLOR_GRAY2BGR)
        frames[100:300, 100:300] = gray_roi_3ch
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),3)


    car_count = len(cars)
    cv2.putText(frames, f'Cars Detected: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if car_count >0:
        cv2.imwrite(f'car_{car_count}.jpg', frames)
    cv2.imshow('Car Detection', frames)

    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()