import csv
import math


def apply_correction(x, y, translation_factor, rotation_angle):
    # Apply translational correction
    corrected_x = x + translation_factor[0]
    corrected_y = y + translation_factor[1]

    # Apply rotational correction
    theta_rad = math.radians(rotation_angle)

    corrected_x_rot = corrected_x * math.cos(theta_rad) - corrected_y * math.sin(theta_rad)
    corrected_y_rot = corrected_x * math.sin(theta_rad) + corrected_y * math.cos(theta_rad)

    return corrected_x_rot, corrected_y_rot


def correct_coordinates(csv_file, translation_factor, rotation_angle):
    corrected_coordinates = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x = float(row[0])
            y = float(row[1])

            corrected_x, corrected_y = apply_correction(x, y, translation_factor, rotation_angle)
            corrected_coordinates.append([corrected_x, corrected_y])

    # Save corrected coordinates to a new CSV file
    output_file = csv_file.split('.')[0] + '_corrected.csv'

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(corrected_coordinates)

    print("Corrected coordinates saved to:", output_file)


# Example usage
csv_file = 'coordinates.csv'
translation_factor = (1.5, 2.0)  # Example translational correction factors
rotation_angle = 45  # Example rotation angle in degrees

correct_coordinates(csv_file, translation_factor, rotation_angle)
