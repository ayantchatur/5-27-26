import cv2

video_source = "cars.mp4"
cascade_path = "cars.xml"

cap = cv2.VideoCapture(0)

car_cascade = cv2.CascadeClassifier(cascade_path)

if car_cascade.empty():
    print("Error loading cascade classifier")
    exit()

while True:
    ret, frames = cap.read()

    if not ret:
        print("cannot read from source")
        break
    
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray,scaleFactor= 2,minNeighbors= 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)

    car_count = len(cars)
    cv2.putText(frames, f'Cars Detected: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Car Detection', frames)

    if cv2.waitKey(33) == 27:
        break

cap.release()
cv2.destroyAllWindows()