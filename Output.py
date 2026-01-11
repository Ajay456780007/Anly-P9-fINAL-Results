import os
import cv2

# Root directory of results
root_dir = "Image_Results/"

# Loop through DB1, DB2, DB3 - fixed duplicate DB2
for db in ["DB1", "DB2", "DB3"]:
    db_path = os.path.join(root_dir, db)
    if not os.path.exists(db_path):
        continue

    # Loop through class folders inside each DB
    for class_name in os.listdir(db_path):
        class_dir = os.path.join(db_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Loop through sample1 and sample2
        for sample_folder in ["sample1", "sample2"]:
            sample_path = os.path.join(class_dir, sample_folder)
            img_path = os.path.join(sample_path, "Original.jpg")
            output_path = os.path.join(sample_path, "Output.jpg")

            if os.path.isfile(img_path):
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue

                # Prepare text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 1  # Start with reasonable thickness
                color = (0, 0, 255)  # Red (BGR)

                # Get image dimensions
                img_height, img_width = img.shape[:2]

                # Target width with margin (90% of image width for better fit)
                target_width = img_width * 0.9
                margin = 5

                # Start with reasonable font scale
                text_size = 1.6
                (text_width, text_height), baseline = cv2.getTextSize(
                    class_name, font, text_size, thickness
                )

                # Dynamically adjust font scale so text fits target width
                while text_width > target_width and text_size > 0.3:
                    text_size -= 0.05
                    (text_width, text_height), baseline = cv2.getTextSize(
                        class_name, font, text_size, thickness
                    )

                # Scale thickness proportionally to font size (max 4px)
                thickness = max(1, min(4, int(text_size * 2)))

                # Position: top-left corner with margin (y accounts for baseline)
                position = (margin, text_height + margin)

                # Put text on image with anti-aliasing
                cv2.putText(
                    img, class_name, position, font, text_size,
                    color, thickness, cv2.LINE_AA
                )

                # Save output image
                cv2.imwrite(output_path, img)
                # print(f"Saved: {output_path} (font_scale={text_size:.2f}, thickness={thickness})")[web:13][web:15]
