import cv2


def detect_track_count(frame, results, labels, class_id):
    """
    Task of the function:
    Draws a box over the detected objects. It writes the score and id on the box.
    This function starts a loop for each detection in `results`. In the loop, the bounding
    box of the detected object is drawn and the class name and confidence score are written
    above the box. If the score of a detected object is below a specified threshold (e.g. score < 0.5)
    or if the detected object does not belong to a desired class (e.g. object), the detection is
    ignored and no drawing is done on it.

    Args:
    - frame: The image on which to draw.
    - results: The detection results from the YOLO model.
    - labels: The list of class labels.
    - class_id: Object id(e.g. 2: 'car')

    The function works in the following steps:
    1. For each detection in `results`, a loop is started using `enumerate`. `enumerate` is both the index
       and the bounding box information of the detected object.
    2. For each detection, the coordinates of the bounding box (`x1`, `y1`, `x2`, `y2`) are converted to integers.
       This is the drawing operations because OpenCV expects the coordinates to be integers.
    3. The confidence score and class ID of the detected object are retrieved. This information is used to determine how much the detected object
       is reliable and which class it belongs to.
    4. If the score of the detected object is below a certain threshold or belongs to an undesirable class, this detection
       no drawing is performed and the next iteration of the loop is started.
    5. For detections with a score above the threshold and belonging to the desired class, the bounding box is drawn and the class name
       and score information is written.

    Details:
    result[0].boxes : stores the coordinates, ID and score of the detected objects (tensor or array object)
    result[0].boxes.xyxy: Contains the coordinates (x1, y1, x2, y2) of the bounding box for each detection.
    result[0].boxes.conf: Contains the confidence score of each detected object (between 0 and 1)
    result[0].boxes.cls: Contains the class ID of each detected object.(2= 'car')
    count: Counts selected objects instantly

    """
    count = 0

    # Loop for each detection object
    # ennumerate:  getting both of them index and value of coordinates
    for i, box in enumerate(results[0].boxes.xyxy):
        # Get the coordinates and convert to integer
        # Coordinates must be int because of no float in OpenCV coordinate list
        x1, y1, x2, y2 = map(int, box[:4])
        # Check it out - print(x2)

        # Get score as float (conf[i])
        score = float(results[0].boxes.conf[i])
        # Check it out - print(score)

        # Get class id as int (conf[i])
        cls = int(results[0].boxes.cls[i])
        # Check it out - print(cls)

        # Get label name
        name = labels[cls]
        # Check it out - print(name)

        if (score > 0.4) and (name == class_id):
            # If detect object(same class_id) increas 1
            count += 1
            # Drow it box(rectangle) on detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Display object ID and Score above box instantly
            text = f'ID:{name}, Score:{score:.2f}'
            cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Display total object number top left on frame
    cv2.putText(frame, f'Total: {count}', (45, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)