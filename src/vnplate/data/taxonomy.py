from __future__ import annotations

ROAD_USER_CLASSES = ("person", "bicycle", "car", "motorbike", "bus", "taxi", "pickup", "truck", "trailer")

VEHICLE_CLASSES = ROAD_USER_CLASSES

ROAD_USER_COLORS = [
    (0, 0, 255),      # person
    (255, 0, 255),    # bicycle
    (50, 90, 180),    # car
    (60, 160, 60),    # bus
    (0, 200, 255),    # taxi
    (180, 60, 60),    # motorbike
    (120, 120, 120),  # pickup
    (60, 60, 200),    # truck
    (200, 120, 40),   # trailer
]

COCO_ROAD_USERS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}

# Default class set for the project (person + bicycle + vehicles)
DEFAULT_CLASSES = ROAD_USER_CLASSES
