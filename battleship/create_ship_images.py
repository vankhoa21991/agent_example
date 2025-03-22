import os
from PIL import Image, ImageDraw

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the images_bateaux directory if it doesn't exist
images_dir = os.path.join(script_dir, "images_bateaux")
os.makedirs(images_dir, exist_ok=True)

# Cell size in pixels - match the size in battleship_game.py
# In battleship_game.py, cote_grille = 400 and there are 10 columns, so each cell is 40 pixels
cell_size = 40

# Ship colors
ship_color = "#3366CC"  # Blue color for ships
border_color = "#1A3366"  # Darker blue for borders

# Create ship images for each size and orientation
for size in [2, 3, 4, 5]:
    # Horizontal ship
    h_width = size * cell_size
    h_height = cell_size
    h_img = Image.new('RGBA', (h_width, h_height), (0, 0, 0, 0))
    h_draw = ImageDraw.Draw(h_img)
    
    # Draw the ship body - make it slightly smaller than the cell to ensure it fits within grid lines
    h_draw.rectangle([(1, 1), (h_width-1, h_height-1)], fill=ship_color, outline=border_color, width=2)
    
    # Add some details to make it look like a ship
    h_draw.rectangle([(h_width-15, 5), (h_width-5, h_height-5)], fill=border_color)  # Control room
    
    # Save the horizontal ship
    h_img.save(os.path.join(images_dir, f"{size}H.png"))
    
    # Vertical ship
    v_width = cell_size
    v_height = size * cell_size
    v_img = Image.new('RGBA', (v_width, v_height), (0, 0, 0, 0))
    v_draw = ImageDraw.Draw(v_img)
    
    # Draw the ship body - make it slightly smaller than the cell to ensure it fits within grid lines
    v_draw.rectangle([(1, 1), (v_width-1, v_height-1)], fill=ship_color, outline=border_color, width=2)
    
    # Add some details to make it look like a ship
    v_draw.rectangle([(5, 5), (v_width-5, 15)], fill=border_color)  # Control room
    
    # Save the vertical ship
    v_img.save(os.path.join(images_dir, f"{size}V.png"))

print("Ship images created successfully!")
