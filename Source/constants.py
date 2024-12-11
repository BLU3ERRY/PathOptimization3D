# constants.py

# Track and Drive Widths (in meters)
TRACK_WIDTH = 8.0
DRIVE_WIDTH = 4.5  # Set narrower than TRACK_WIDTH for safety margins

# Number of sectors
N_SECTORS = 600
N_OPTIMIZABLE_SECTORS = N_SECTORS - 2  # Exclude first and last sectors

# PSO Parameters
PSO_N_PARTICLES = 1500
PSO_N_ITERATIONS = 100
INERTIA_WEIGHT = 0.5
COGNITIVE_PARAM = 1.5
SOCIAL_PARAM = 1.5

# G-descent Parameters
GDESC_N_ITERATIONS = 30
LEARNING_RATE = 0.1
DELTA = 0.01

# Vehicle Specifications
VEHICLE_LENGTH = 2.1        # meters
VEHICLE_WIDTH = 1.2         # meters
VEHICLE_HEIGHT = 0.95       # meters
MAX_SPEED_KMH = 20.0        # km/h
MAX_SPEED = MAX_SPEED_KMH / 3.6  # Convert to m/s
STEERING_RANGE = (-20, 20)  # degrees
CENTER_TO_FRONT = 0.5       # meters
CENTER_TO_REAR = 0.523      # meters
WHEEL_SPACING = 1.0         # meters
AXLE_SPACING = 1.023        # meters
