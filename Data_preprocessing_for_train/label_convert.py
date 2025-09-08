import os
import math
import numpy as np

def convert_file(input_path, output_path, scale_factor=1.05):
    """
    Convert a single file from the original format to the new format.
    Original: cx cy cz dx dy dz heading
    New: cx cy cz dx dy dz heading_rad Liftingobjects
    Also scales the geometric dimensions (dx, dy, dz) by scale_factor
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    converted_lines = []
    for line in lines:
        # Split the line into values
        values = line.strip().split()
        if len(values) != 7:
            print(f"Warning: Skipping invalid line in {input_path}: {line}")
            continue
            
        # Convert values to float
        try:
            cx, cy, cz, dx, dy, dz, heading = map(float, values)
        except ValueError:
            print(f"Warning: Could not convert values in {input_path}: {line}")
            continue
            
        # Scale the geometric dimensions
        dx *= scale_factor
        dy *= scale_factor
        dz *= scale_factor
            
        # Convert heading to radians
        heading_rad = math.radians(heading)
        
        # Create new line with the converted format and scaled dimensions
        new_line = f"{cx:.6f} {cy:.6f} {cz:.6f} {dx:.6f} {dy:.6f} {dz:.6f} {heading_rad:.6f} Liftingobjects\n"
        converted_lines.append(new_line)
    
    # Write converted lines to output file
    with open(output_path, 'w') as f:
        f.writelines(converted_lines)

def batch_convert_files(input_folder, output_folder, scale_factor=1.05):
    """
    Batch convert all txt files in the input folder to the new format
    and save them in the output folder.
    Also scales the geometric dimensions by scale_factor.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all txt files in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                convert_file(input_path, output_path, scale_factor)
                print(f"Successfully converted and scaled: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Example usage
if __name__ == "__main__":
    input_folder = "raw_label_from_unity"  # 替换为输入文件夹路径
    output_folder = "label_for_training"  # 替换为输出文件夹路径
    
    scale_factor = 1.05  # 3D框几何尺寸的缩放因子
    batch_convert_files(input_folder, output_folder, scale_factor)
