import os

# Folder where images are saved
output_dir = 'car_images'

for f in os.listdir(output_dir):
    if f.endswith(('.jpg', '.jpeg', '.png')):
        name, ext = os.path.splitext(f)
        # Remove leading zeros
        new_name = name.lstrip('0') + ext
        os.rename(os.path.join(output_dir, f), os.path.join(output_dir, new_name))

print("Leading zeros removed!")
