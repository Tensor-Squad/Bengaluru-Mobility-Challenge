from predict import predict_counts
from ultralytics import YOLO
import cv2
import numpy as np
import json
import sys

# (For Docker only) Load input and output paths from command-line arguments
input_json_path = sys.argv[1]
output_json_path = sys.argv[2]

# Constants for configuration
CONFIG_PATH = 'config.json'
TEMPLATE_PATH = 'template.json'
CONFIDENCE_THRESHOLD = 0.65

# Load configuration and input data
input_data = json.load(open(input_json_path, 'r'))
config = json.load(open(CONFIG_PATH, 'r'))
template = json.load(open(TEMPLATE_PATH, 'r'))

# Load the YOLOv8 model
model = YOLO(config['model_path'])

target_classes = config['target_classes']

# Initialize data structures
object_regions = {}
tracker_history = {}
class_patterns = {cls: {pattern: 0 for pattern in config[next(iter(input_data))]['patterns']} for cls in target_classes}
class_names = {0: 'Bicycle', 1: 'Bus', 2: 'Car', 3: 'LCV', 4: 'Three Wheeler', 5: 'Truck', 6: 'Two Wheeler'}


# Function to get the region of a point
def get_region(x, y, regions):
    """Determine which region the point (x, y) belongs to by checking if it's inside any polygon."""
    point = (x, y)
    for region_name, polygon in regions.items():
        polygon_np = np.array(polygon, np.int32)
        if cv2.pointPolygonTest(polygon_np, point, False) >= 0:
            return region_name
    return ''


# Function to update the object regions
def update_object_regions(track_id, center_x, center_y, class_id, regions, timestamp):
    """Update the regions and patterns for each object."""
    current_region = get_region(center_x, center_y, regions)
    if current_region == '':
        return
    if track_id not in object_regions:
        object_regions[track_id] = {
            'regions': [current_region],
            'class_id_counts': {class_id: 1},
            'first_appearance': timestamp
        }
    else:
        if current_region != object_regions[track_id]['regions'][-1]:
            object_regions[track_id]['regions'].append(current_region)
        if class_id in object_regions[track_id]['class_id_counts']:
            object_regions[track_id]['class_id_counts'][class_id] += 1
        else:
            object_regions[track_id]['class_id_counts'][class_id] = 1


# Function to get the most frequent class ID
def get_mode_class_id(class_id_counts):
    """Return the most frequent class_id from the dictionary."""
    if not class_id_counts:
        return None
    # Find the class ID with the highest count
    mode_class_id = max(class_id_counts, key=class_id_counts.get)

    return mode_class_id


# Function to process each frame
def process_frame(frame, regions, timestamp):
    """Process each frame to detect and track objects."""
    results = model.track(frame, classes=target_classes, conf=CONFIDENCE_THRESHOLD, persist=True)
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
    class_ids = results[0].boxes.cls.int().cpu().tolist() if results[0].boxes.cls is not None else []

    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        x, y, _, _ = box
        center_x, center_y = float(x), float(y)
        update_object_regions(track_id, center_x, center_y, class_id, regions, timestamp)


# Functions to rename specific patterns
def process_counts(template, camera_id, key1, key2, count_type):
    if key1 in template[camera_id][count_type]:
        template[camera_id][count_type][key2] = template[camera_id][count_type][key1].copy()
        for key in template[camera_id][count_type][key1]:
            template[camera_id][count_type][key1][key] = 0


def process_key_pairs(camera_id, key_pairs, template):
    for key1, key2 in key_pairs:
        process_counts(template, camera_id, key1, key2, "Cumulative Counts")
        process_counts(template, camera_id, key1, key2, "Predicted Counts")


# Function to process counts for specific keys
def process_counts_for_keys(template, camera_id, source_key, target_key, count_type):
    for key in template[camera_id][count_type][source_key]:
        template[camera_id][count_type][target_key][key] += template[camera_id][count_type][source_key][key]
        template[camera_id][count_type][source_key][key] = 0


# Function to print the class patterns and save to JSON
def print_class_patterns(camera_id, patterns):
    """Print the movement patterns of each object and save to JSON."""

    # Replace the dummy cam_id with the actual camera_id
    if "Cam_ID" in template:
        template[camera_id] = template.pop("Cam_ID")

    if camera_id not in template:
        template[camera_id] = {"Cumulative Counts": {}, "Predicted Counts": {}}

    for obj_id, data in object_regions.items():
        regions = data['regions']
        class_ids = data['class_id_counts']
        class_id = get_mode_class_id(class_ids)
        class_name = class_names.get(class_id, 'Unknown')  # Get the class name or 'Unknown' if not found
        if len(regions) >= 2:
            first_region = regions[0]
            last_region = regions[-1]
            pattern_key = f"{first_region}{last_region}"
            if pattern_key in class_patterns[class_id]:
                class_patterns[class_id][pattern_key] += 1
            if pattern_key in patterns:
                tracker_id = obj_id  # Assuming obj_id is the tracker_id
                timestamp = data['first_appearance']  # Assuming timestamp is stored in data
                tracker_history[tracker_id] = {
                    'timestamp': timestamp,
                    'class_id': class_name,  # Replace class_id with class_name
                    'pattern': pattern_key
                }

    # save the predicted counts
    predictions = predict_counts(tracker_history)
    for pattern, class_counts in predictions.items():
        if pattern not in template[camera_id]["Predicted Counts"]:
            template[camera_id]["Predicted Counts"][pattern] = {
                "Bicycle": 0, "Bus": 0, "Car": 0, "LCV": 0, "Three Wheeler": 0, "Truck": 0, "Two Wheeler": 0
            }
        for class_name, count in class_counts.items():
            template[camera_id]["Predicted Counts"][pattern][class_name] = count

    # save the cumulative counts
    for class_id, patterns in class_patterns.items():
        class_name = class_names.get(class_id, f"Unknown({class_id})")

        # print the count ( Debugging )
        total_patterns = sum(patterns.values())
        pattern_strings = [f"{pattern}: {count}" for pattern, count in patterns.items() if count > 0]
        if pattern_strings:
            print(f"Class {class_name} (Total: {total_patterns}): {', '.join(pattern_strings)}")
        else:
            print(f"Class {class_name} (Total: {total_patterns}): No patterns detected")

        # save in the required format
        for pattern, count in patterns.items():
            if pattern not in template[camera_id]["Cumulative Counts"]:
                template[camera_id]["Cumulative Counts"][pattern] = {
                    "Bicycle": 0, "Bus": 0, "Car": 0, "LCV": 0, "Three Wheeler": 0, "Truck": 0, "Two Wheeler": 0
                }
            template[camera_id]["Cumulative Counts"][pattern][class_name] = count

    # Some camera view specific processing
    # Define key pairs for each camera_id
    key_pairs_stn_hd_1 = [("AC", "BC"), ("AE", "BE"), ("CE", "DE"), ("CA", "DA"), ("EA", "FA"), ("EC", "FC")]
    key_pairs_ms_ramaiah_jn_fix_2 = [
        ("AC", "BC"), ("AE", "BE"), ("AG", "BG"), ("CA", "DA"), ("CE", "DE"),
        ("CG", "DG"), ("GA", "HA"), ("GC", "HC"), ("GE", "HE")
    ]

    if camera_id == "Stn_HD_1":
        process_key_pairs(camera_id, key_pairs_stn_hd_1, template)

    if camera_id == "MS_Ramaiah_JN_FIX_2":
        process_key_pairs(camera_id, key_pairs_ms_ramaiah_jn_fix_2, template)

        # Process Cumulative Counts
        process_counts_for_keys(template, camera_id, "CF", "DG", "Cumulative Counts")
        process_counts_for_keys(template, camera_id, "FE", "HE", "Cumulative Counts")

        # Process Predicted Counts
        process_counts_for_keys(template, camera_id, "CF", "DG", "Predicted Counts")
        process_counts_for_keys(template, camera_id, "FE", "HE", "Predicted Counts")

    json.dump(template, open(output_json_path, 'w'), indent=4)


# Initialize cumulative timestamp
cumulative_timestamp = 0

for camera_id, videos in input_data.items():
    camera_config = config[camera_id]
    regions = {key: tuple(value) for key, value in camera_config['regions'].items()}
    patterns = camera_config['patterns']

    for video_id, video_path in videos.items():
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        video_timestamp = 0  # Initialize video-specific timestamp

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    timestamp = int(cap.get(
                        cv2.CAP_PROP_POS_MSEC) / 1000) + cumulative_timestamp  # Add cumulative timestamp
                    process_frame(frame, regions, timestamp)
                    video_timestamp = int(
                        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)  # Update video-specific timestamp in seconds
                else:
                    break
        finally:
            # Update cumulative timestamp with the ending timestamp of the current video
            cumulative_timestamp += video_timestamp
            cap.release()

    print_class_patterns(camera_id, patterns)
