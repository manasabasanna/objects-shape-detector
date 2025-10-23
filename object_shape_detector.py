import cv2
import numpy as np
import os
from datetime import datetime


# Function to setup Manasa folder with exact path
def setup_manasa_folder():
    """Use the exact Manasa folder path from your file"""
    exact_path = r"C:\Users\Manasa\Pictures"  # Your exact Manasa folder path

    if os.path.exists(exact_path):
        print(f"✅ Found your Manasa folder: {exact_path}")
        return exact_path
    else:
        print(f"❌ Manasa folder not found: {exact_path}")
        print("📁 Creating it now...")
        os.makedirs(exact_path, exist_ok=True)
        return exact_path


# Function to safely destroy windows
def safe_destroy_window(window_name):
    """Safely destroy a window if it exists"""
    try:
        cv2.destroyWindow(window_name)
    except:
        pass  # Window doesn't exist, which is fine


# Function to detect shape based on contour
def detect_shape(contour):
    shape = "Unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    num_vertices = len(approx)

    if num_vertices == 3:
        shape = "Triangle"
    elif num_vertices == 4:
        # Distinguish square vs rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif num_vertices == 5:
        shape = "Pentagon"
    elif num_vertices == 6:
        shape = "Hexagon"
    else:
        # For circles/ovals, calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            shape = "Circle" if circularity > 0.7 else "Oval"

    return shape, approx


# Function to convert pixels to real-world measurements
def convert_to_cm(pixels, reference_pixels, reference_cm):
    """Convert pixel measurements to centimeters"""
    return (pixels / reference_pixels) * reference_cm


# Function to save captured photo with metadata
def save_photo_with_metadata(frame, contours, object_count, save_dir):
    """Save photo with timestamp and detection metadata"""
    # Create a copy of the frame to avoid modifying original
    photo = frame.copy()

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(save_dir, f"shape_detection_{timestamp}.jpg")

    # Add timestamp and count info to the photo
    cv2.putText(photo, f"Captured: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add object count summary
    count_y = 60
    total_objects = sum(object_count.values())
    cv2.putText(photo, f"Total Objects: {total_objects}", (10, count_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save the photo
    success = cv2.imwrite(filename, photo)

    if success:
        # Save metadata to text file
        metadata_filename = os.path.join(save_dir, f"shape_detection_{timestamp}.txt")
        with open(metadata_filename, 'w') as f:
            f.write(f"Shape Detection Capture - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total objects detected: {total_objects}\n\n")
            f.write("Object Count Breakdown:\n")
            for shape_type, count in object_count.items():
                if count > 0:
                    f.write(f"  {shape_type}: {count}\n")

            f.write(f"\nContours found: {len(contours)}\n")
            f.write(f"Image resolution: {frame.shape[1]}x{frame.shape[0]}\n")
            f.write(f"Calibration: {REFERENCE_WIDTH_CM}cm = {REFERENCE_WIDTH_PX}px\n")

        print(f"✅ Photo and metadata saved: {filename}")
        return filename
    else:
        print(f"❌ Failed to save photo: {filename}")
        return None


# Function to display existing photos from Manasa Pictures folder
def show_manasa_photos(manasa_folder):
    """Display all photos from Manasa Pictures folder"""
    print(f"📂 Looking for photos in: {manasa_folder}")

    # Get all image files (multiple formats)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    try:
        all_files = os.listdir(manasa_folder) if os.path.exists(manasa_folder) else []
        photo_files = []
        for file in all_files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                photo_files.append(file)

        print(f"🔍 Found {len(photo_files)} image files in your Pictures folder")

        # Show what files were found
        if photo_files:
            print("📷 Images found:")
            for img in photo_files[:10]:  # Show first 10 files
                print(f"   - {img}")
            if len(photo_files) > 10:
                print(f"   ... and {len(photo_files) - 10} more")

    except Exception as e:
        print(f"❌ Error reading folder: {e}")
        photo_files = []

    if not photo_files:
        # Show helpful message with instructions
        message_window = np.zeros((400, 700, 3), dtype=np.uint8)

        cv2.putText(message_window, "NO PHOTOS FOUND IN YOUR PICTURES FOLDER", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(message_window, f"Folder: {manasa_folder}", (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(message_window, "To add photos:", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(message_window, "1. Press 'p' in main window to capture new photos", (70, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(message_window, "2. Or manually copy images to the folder above", (70, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(message_window, "3. Then press 'g' again to view them", (70, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(message_window, "Press any key to return to live view", (200, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("No Photos Found", message_window)
        cv2.waitKey(0)
        safe_destroy_window("No Photos Found")
        return

    # Sort by modification time (newest first)
    try:
        photo_files.sort(key=lambda x: os.path.getmtime(os.path.join(manasa_folder, x)), reverse=True)
        photos_to_show = photo_files[:9]  # Show up to 9 photos in grid
    except:
        photos_to_show = photo_files[:9]

    # Create gallery window
    gallery_height = 800
    gallery_width = 1000
    gallery_window = np.zeros((gallery_height, gallery_width, 3), dtype=np.uint8)

    # Display gallery header
    cv2.putText(gallery_window, f"MANASA PICTURES FOLDER GALLERY",
                (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(gallery_window, f"Showing {len(photos_to_show)} of {len(photo_files)} photos",
                (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(gallery_window, f"Folder: {manasa_folder}",
                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(gallery_window, "Press any key to return to live view",
                (50, gallery_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display photos in 3x3 grid
    thumb_size = (250, 180)
    margin = 20
    start_x, start_y = 50, 120

    photos_displayed = 0

    for i, photo_file in enumerate(photos_to_show):
        try:
            full_path = os.path.join(manasa_folder, photo_file)
            print(f"🖼️  Loading: {photo_file}")
            img = cv2.imread(full_path)

            if img is not None:
                # Resize to thumbnail
                thumb = cv2.resize(img, thumb_size)

                # Calculate grid position (3x3)
                row = i // 3
                col = i % 3

                x_pos = start_x + col * (thumb_size[0] + margin)
                y_pos = start_y + row * (thumb_size[1] + margin)

                # Place thumbnail if it fits
                if y_pos + thumb_size[1] < gallery_height - 50 and x_pos + thumb_size[0] < gallery_width:
                    gallery_window[y_pos:y_pos + thumb_size[1], x_pos:x_pos + thumb_size[0]] = thumb

                    # Add filename
                    display_name = photo_file[:20] + "..." if len(photo_file) > 20 else photo_file
                    cv2.putText(gallery_window, display_name,
                                (x_pos, y_pos + thumb_size[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                    photos_displayed += 1
                    print(f"✅ Successfully displayed: {photo_file}")
                else:
                    print(f"⚠️  Skipping {photo_file} - out of grid space")
            else:
                print(f"❌ Failed to load: {photo_file}")

        except Exception as e:
            print(f"❌ Error loading {photo_file}: {e}")

    if photos_displayed > 0:
        cv2.imshow("Photo Gallery", gallery_window)
        cv2.waitKey(0)
        safe_destroy_window("Photo Gallery")
    else:
        # Show error if no photos could be loaded
        error_window = np.zeros((300, 600, 3), dtype=np.uint8)
        cv2.putText(error_window, "ERROR LOADING PHOTOS", (150, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(error_window, "Check console for details", (150, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(error_window, "Press any key to return", (180, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Gallery Error", error_window)
        cv2.waitKey(0)
        safe_destroy_window("Gallery Error")


# Calibration settings
REFERENCE_WIDTH_CM = 5.0
REFERENCE_WIDTH_PX = 100

# Setup Manasa folder with exact path
MANASA_FOLDER = setup_manasa_folder()

# Start webcam
cap = cv2.VideoCapture(0)

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Object counting dictionary
object_count = {
    "Triangle": 0,
    "Square": 0,
    "Rectangle": 0,
    "Circle": 0,
    "Pentagon": 0,
    "Hexagon": 0,
    "Oval": 0,
    "Other": 0
}

# Colors for different shapes
shape_colors = {
    "Triangle": (0, 255, 0),  # Green
    "Square": (255, 0, 0),  # Blue
    "Rectangle": (255, 255, 0),  # Cyan
    "Circle": (0, 0, 255),  # Red
    "Pentagon": (255, 0, 255),  # Magenta
    "Hexagon": (0, 255, 255),  # Yellow
    "Oval": (128, 0, 128),  # Purple
    "Other": (255, 255, 255)  # White
}

# Photo capture counter
photo_counter = 0
max_photos = 1000

print("=" * 60)
print("🎯 SHAPE DETECTION - CONNECTED TO YOUR PICTURES FOLDER")
print("=" * 60)
print(f"📁 Using your exact folder: {MANASA_FOLDER}")
print("\n=== CONTROLS ===")
print("📸 Press 'p' - Capture photo with shape detection")
print("🖼️  Press 'g' - View ALL images in your Pictures folder")
print("🔄 Press 'r' - Reset object counters")
print("📐 Press 'c' - Calibrate reference object")
print("❌ Press 'q' - Quit application")
print("\n💡 The gallery will show ALL images from your Pictures folder!")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Create a copy for display
    display_frame = frame.copy()

    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use adaptive threshold for better handling of lighting variations
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Reset counts for this frame (we'll recount)
    current_frame_count = {key: 0 for key in object_count.keys()}

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter out small noise
        if area < 1500:
            continue

        shape, approx = detect_shape(cnt)
        x, y, w, h = cv2.boundingRect(approx)

        # Update count for current frame
        current_frame_count[shape] = current_frame_count.get(shape, 0) + 1

        # Calculate real-world measurements
        width_cm = convert_to_cm(w, REFERENCE_WIDTH_PX, REFERENCE_WIDTH_CM)
        height_cm = convert_to_cm(h, REFERENCE_WIDTH_PX, REFERENCE_WIDTH_CM)
        area_cm2 = width_cm * height_cm

        # Calculate perimeter
        perimeter = cv2.arcLength(cnt, True)
        perimeter_cm = convert_to_cm(perimeter, REFERENCE_WIDTH_PX, REFERENCE_WIDTH_CM)

        # Get color for this shape
        color = shape_colors.get(shape, (255, 255, 255))

        # Draw contour and bounding box
        cv2.drawContours(display_frame, [approx], 0, color, 2)
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

        # Display shape information
        cv2.putText(display_frame, f"Shape: {shape}", (x, y - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display pixel measurements
        cv2.putText(display_frame, f"Pixels: {w}x{h}", (x, y - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Area: {int(area)} px", (x, y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display real-world measurements
        cv2.putText(display_frame, f"Size: {width_cm:.1f}x{height_cm:.1f} cm", (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Area: {area_cm2:.1f} cm2", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(display_frame, f"Perimeter: {perimeter_cm:.1f} cm", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Update object counts (only if object is still present)
    for shape_type in object_count.keys():
        if current_frame_count[shape_type] > 0:
            object_count[shape_type] = current_frame_count[shape_type]

    # Display object counting statistics
    stats_y = 30
    cv2.putText(display_frame, "OBJECT COUNT:", (10, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    stats_y += 25

    for shape_type, count in object_count.items():
        if count > 0:  # Only show shapes that are detected
            color = shape_colors.get(shape_type, (255, 255, 255))
            cv2.putText(display_frame, f"{shape_type}: {count}", (10, stats_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            stats_y += 20

    # Display folder info and controls
    cv2.putText(display_frame, f"Folder: Manasa/Pictures",
                (10, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Photos Captured: {photo_counter}",
                (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Reference: {REFERENCE_WIDTH_CM}cm = {REFERENCE_WIDTH_PX}px",
                (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(display_frame, "p=Photo  g=Gallery  r=Reset  c=Calibrate  q=Quit",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the frames
    cv2.imshow("Shape Detection - Connected to Your Pictures", display_frame)
    cv2.imshow("Threshold View", thresh)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        object_count = {key: 0 for key in object_count.keys()}
        print("🔄 Object counters reset!")
    elif key == ord('c'):
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            _, approx = detect_shape(largest_contour)
            x, y, w, h = cv2.boundingRect(approx)
            REFERENCE_WIDTH_PX = w
            print(f"📐 Calibrated! Reference: {w} pixels = {REFERENCE_WIDTH_CM}cm")
    elif key == ord('p'):
        if photo_counter < max_photos:
            filename = save_photo_with_metadata(display_frame, contours, current_frame_count, MANASA_FOLDER)
            if filename:
                photo_counter += 1
                print(f"📸 Photo {photo_counter} captured in your Pictures folder!")

                # Show confirmation
                confirmation_frame = display_frame.copy()
                cv2.putText(confirmation_frame, "✅ PHOTO SAVED TO YOUR PICTURES!",
                            (confirmation_frame.shape[1] // 2 - 200, confirmation_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Shape Detection - Connected to Your Pictures", confirmation_frame)
                cv2.waitKey(500)
        else:
            print(f"❌ Photo limit reached ({max_photos})")
    elif key == ord('g'):
        print(f"🖼️  Opening your Pictures folder gallery...")
        safe_destroy_window("Shape Detection - Connected to Your Pictures")
        safe_destroy_window("Threshold View")
        show_manasa_photos(MANASA_FOLDER)
        print("🔄 Returning to live view...")

cap.release()
safe_destroy_window("Shape Detection - Connected to Your Pictures")
safe_destroy_window("Threshold View")
cv2.destroyAllWindows()

# Final summary
print(f"\n" + "=" * 50)
print("🎉 SESSION COMPLETED!")
print(f"📊 Total photos captured: {photo_counter}")
print(f"📁 All photos saved in your Pictures folder: {MANASA_FOLDER}")
if photo_counter > 0:
    print("💡 Press 'g' next time to see your captured photos!")
print("=" * 50)