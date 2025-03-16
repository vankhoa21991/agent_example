import tkinter as tk
from tkinter import messagebox
import random

# Constants for the game
GRID_SIZE = 10
CELL_SIZE = 40
SHIP_SIZES = {
    'Carrier': 5,
    'Battleship': 4,
    'Cruiser': 3,
    'Submarine': 3,
    'Destroyer': 2
}
COLORS = {
    'water': '#ADD8E6',  # Light blue
    'ship': '#808080',   # Gray
    'hit': '#FF0000',    # Red
    'miss': '#FFFFFF',   # White
    'background': '#F0F0F0'  # Light gray
}

# Global variables
player_board = []
computer_board = []
player_ships = {}
computer_ships = {}
game_over = False
shots_fired = 0
hits = 0

# Global variable for cheat mode
cheat_mode = False

def initialize_board():
    """Create an empty board filled with water."""
    return [['water' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

def place_ships_randomly(board, ships):
    """Place ships randomly on the board."""
    for ship_name, ship_size in SHIP_SIZES.items():
        placed = False
        while not placed:
            # Randomly choose orientation (0 for horizontal, 1 for vertical)
            orientation = random.randint(0, 1)
            
            if orientation == 0:  # Horizontal
                x = random.randint(0, GRID_SIZE - ship_size)
                y = random.randint(0, GRID_SIZE - 1)
                
                # Check if the position is valid
                valid = True
                for i in range(ship_size):
                    if board[y][x + i] != 'water':
                        valid = False
                        break
                
                # Place the ship if the position is valid
                if valid:
                    ships[ship_name] = []
                    for i in range(ship_size):
                        board[y][x + i] = 'ship'
                        ships[ship_name].append((y, x + i))
                    placed = True
            else:  # Vertical
                x = random.randint(0, GRID_SIZE - 1)
                y = random.randint(0, GRID_SIZE - ship_size)
                
                # Check if the position is valid
                valid = True
                for i in range(ship_size):
                    if board[y + i][x] != 'water':
                        valid = False
                        break
                
                # Place the ship if the position is valid
                if valid:
                    ships[ship_name] = []
                    for i in range(ship_size):
                        board[y + i][x] = 'ship'
                        ships[ship_name].append((y + i, x))
                    placed = True

def draw_board(canvas, board, is_player_board):
    """Draw the game board on the canvas."""
    canvas.delete("all")

    # Draw grid
    for i in range(GRID_SIZE + 1):
        # Draw horizontal lines
        canvas.create_line(0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE, i * CELL_SIZE, fill="#888888")
        # Draw vertical lines
        canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, GRID_SIZE * CELL_SIZE, fill="#888888")

    # Add row labels (1 to 10)
    for i in range(GRID_SIZE):
        row_label = str(i + 1)  # 1, 2, 3, ...
        canvas.create_text(
            (i + 0.5) * CELL_SIZE, 20,  # Position above the first row
            text=row_label,
            font=("Arial", 12),
            fill="black",
            anchor="center"
        )

    # Add column labels (A to J)
    for i in range(GRID_SIZE):
        col_label = chr(65 + i)  # A, B, C, ...
        canvas.create_text(
            20, (i + 0.5) * CELL_SIZE,  # Position to the left of the first column
            text=col_label,
            font=("Arial", 12),
            fill="black",
            anchor="center"
        )
    
    # Draw cells
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cell_state = board[y][x]
            
            # For computer's board, don't show ships that haven't been hit
            # unless cheat mode is active
            if not is_player_board and cell_state == 'ship' and not cheat_mode:
                cell_color = COLORS['water']
            else:
                cell_color = COLORS[cell_state]
            
            canvas.create_rectangle(
                x * CELL_SIZE, y * CELL_SIZE,
                (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                fill=cell_color, outline='black'
            )

def computer_turn():
    """Computer takes a shot at the player's board."""
    global game_over
    
    if game_over:
        return
    
    # Randomly select a position that hasn't been shot at
    valid_shot = False
    while not valid_shot:
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        
        # Check if this position hasn't been shot at yet
        if player_board[y][x] == 'water' or player_board[y][x] == 'ship':
            valid_shot = True
    
    # Process the shot
    if player_board[y][x] == 'ship':
        player_board[y][x] = 'hit'
        
        # Check if a ship is sunk
        for ship_name, positions in player_ships.items():
            if (y, x) in positions:
                positions.remove((y, x))
                if not positions:  # Ship is sunk
                    messagebox.showinfo("Ship Sunk", f"Computer sunk your {ship_name}!")
                break
        
        # Check if all player ships are sunk
        all_sunk = True
        for positions in player_ships.values():
            if positions:
                all_sunk = False
                break
        
        if all_sunk:
            game_over = True
            messagebox.showinfo("Game Over", "Computer wins! All your ships are sunk.")
    else:
        player_board[y][x] = 'miss'
    
    # Update the player's board display
    draw_board(player_canvas, player_board, True)

def player_shot(event):
    """Process player's shot on the computer's board."""
    global game_over, shots_fired, hits
    
    if game_over:
        return
    
    # Get the cell coordinates from the event
    x = event.x // CELL_SIZE
    y = event.y // CELL_SIZE
    
    # Check if the coordinates are valid
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        return
    
    # Check if this cell has already been shot at
    if computer_board[y][x] == 'hit' or computer_board[y][x] == 'miss':
        return
    
    # Process the shot
    shots_fired += 1
    if computer_board[y][x] == 'ship':
        computer_board[y][x] = 'hit'
        hits += 1
        
        # Check if a ship is sunk
        for ship_name, positions in computer_ships.items():
            if (y, x) in positions:
                positions.remove((y, x))
                if not positions:  # Ship is sunk
                    messagebox.showinfo("Ship Sunk", f"You sunk the computer's {ship_name}!")
                break
        
        # Check if all computer ships are sunk
        all_sunk = True
        for positions in computer_ships.values():
            if positions:
                all_sunk = False
                break
        
        if all_sunk:
            game_over = True
            accuracy = (hits / shots_fired) * 100
            messagebox.showinfo("Game Over", f"You win! All enemy ships are sunk.\nShots fired: {shots_fired}\nHits: {hits}\nAccuracy: {accuracy:.1f}%")
    else:
        computer_board[y][x] = 'miss'
    
    # Update the computer's board display
    draw_board(computer_canvas, computer_board, False)
    
    # Computer's turn
    if not game_over:
        computer_turn()

def reset_game():
    """Reset the game to its initial state."""
    global player_board, computer_board, player_ships, computer_ships, game_over, shots_fired, hits
    
    # Reset game variables
    player_board = initialize_board()
    computer_board = initialize_board()
    player_ships = {}
    computer_ships = {}
    game_over = False
    shots_fired = 0
    hits = 0
    
    # Place ships
    place_ships_randomly(player_board, player_ships)
    place_ships_randomly(computer_board, computer_ships)
    
    # Update board displays
    draw_board(player_canvas, player_board, True)
    draw_board(computer_canvas, computer_board, False)
    
    # Update status
    status_label.config(text="Game started! Click 'New Game' to start.")

def toggle_cheat():
    """Toggle cheat mode to show/hide enemy ships."""
    global cheat_mode
    cheat_mode = not cheat_mode
    
    # Update the computer's board display to show/hide ships
    draw_board(computer_canvas, computer_board, False)
    
    # Update the cheat button text
    if cheat_mode:
        cheat_button.config(text="Hide Enemy Ships")
        status_label.config(text="CHEAT MODE ACTIVE: Enemy ships are visible!")
    else:
        cheat_button.config(text="Show Enemy Ships")
        status_label.config(text="Cheat mode deactivated.")

def create_game_window():
    """Create the main game window and UI elements."""
    global player_canvas, computer_canvas, status_label, cheat_button, root

    # Create main window
    root = tk.Tk()
    root.title("Battleship Game")
    root.configure(bg=COLORS['background'])

    # Create frames for the boards
    player_frame = tk.Frame(root, padx=10, pady=10, bg=COLORS['background'])
    player_frame.grid(row=0, column=0)

    computer_frame = tk.Frame(root, padx=10, pady=10, bg=COLORS['background'])
    computer_frame.grid(row=0, column=1)

    # Create menu frame on the right side
    menu_frame = tk.Frame(root, padx=10, pady=10, bg=COLORS['background'], width=200)
    menu_frame.grid(row=0, column=2, rowspan=2, sticky="ns")

    # Add a title for the menu
    tk.Label(menu_frame, text="Game Menu", font=("Arial", 16, "bold"), bg=COLORS['background']).pack(pady=(0, 20))

    # Add cheat button to the menu
    cheat_button = tk.Button(
        menu_frame,
        text="Show Enemy Ships",
        command=toggle_cheat,
        font=("Arial", 12),
        width=15,
        height=2,
        bg="#f0f0f0"
    )
    cheat_button.pack(pady=10)

    # Create reset button
    reset_button = tk.Button(menu_frame, text="New Game", command=reset_game, font=("Arial", 12), width=15, height=2, bg="#f0f0f0")
    reset_button.pack(pady=10)

    # # Create status label
    # status_label = tk.Label(
    #     menu_frame,
    #     text="Welcome to Battleship! Click 'New Game' to start.",
    #     font=("Arial", 12),
    #     bg=COLORS['background'],
    #     width=20,
    #     wraplength=180
    # )
    # status_label.pack(pady=10)

    # # Create instructions
    # instructions = """
    # Instructions:
    # 1. Click 'New Game' to start a new game
    # 2. Your ships are shown on the left grid
    # 3. Click on the right grid to fire at the enemy
    # 4. Sink all enemy ships to win!
    # """

    # tk.Label(
    #     menu_frame,
    #     text=instructions,
    #     justify=tk.LEFT,
    #     font=("Arial", 10),
    #     bg=COLORS['background'],
    #     width=20,
    #     wraplength=180
    # ).pack(pady=10)

    # Create labels for the boards
    tk.Label(player_frame, text="Your Fleet", font=("Arial", 14), bg=COLORS['background']).pack()
    tk.Label(computer_frame, text="Enemy Fleet", font=("Arial", 14), bg=COLORS['background']).pack()

    # Create canvases for the boards
    player_canvas = tk.Canvas(
        player_frame,
        width=GRID_SIZE * CELL_SIZE,
        height=GRID_SIZE * CELL_SIZE,
        bg=COLORS['water']
    )
    player_canvas.pack()

    computer_canvas = tk.Canvas(
        computer_frame,
        width=GRID_SIZE * CELL_SIZE,
        height=GRID_SIZE * CELL_SIZE,
        bg=COLORS['water']
    )
    computer_canvas.pack()

    # Bind click event to computer's canvas
    computer_canvas.bind("<Button-1>", player_shot)

    # Create control frame
    control_frame = tk.Frame(root, padx=10, pady=10, bg=COLORS['background'])
    control_frame.grid(row=1, column=0, columnspan=3)


    # Create legend frame
    legend_frame = tk.Frame(root, padx=10, pady=10, bg=COLORS['background'])
    legend_frame.grid(row=2, column=0, columnspan=3)

    # Create legend items
    legend_items = [
        ("Water", COLORS['water']),
        ("Ship", COLORS['ship']),
        ("Hit", COLORS['hit']),
        ("Miss", COLORS['miss'])
    ]

    for i, (text, color) in enumerate(legend_items):
        frame = tk.Frame(legend_frame, padx=5, bg=COLORS['background'])
        frame.grid(row=0, column=i, padx=10)

        canvas = tk.Canvas(frame, width=20, height=20, bg=color)
        canvas.pack()

        label = tk.Label(frame, text=text, bg=COLORS['background'])
        label.pack()

    # # Create instructions
    # instructions = """
    # Instructions:
    # 1. Click 'New Game' to start a new game
    # 2. Your ships are shown on the left grid
    # 3. Click on the right grid to fire at the enemy
    # 4. Sink all enemy ships to win!
    # """

    # tk.Label(
    #     root,
    #     text=instructions,
    #     justify=tk.LEFT,
    #     font=("Arial", 10),
    #     bg=COLORS['background']
    # ).grid(row=3, column=0, columnspan=3, padx=10, pady=10)

    return root

def main():
    """Main function to start the game."""
    global player_board, computer_board, player_ships, computer_ships, root, player_canvas, computer_canvas, status_label, cheat_button

    # Initialize boards
    player_board = initialize_board()
    computer_board = initialize_board()

    # Create game window
    root = create_game_window()

    # Place ships
    place_ships_randomly(computer_board, computer_ships)

    # Draw initial boards
    draw_board(player_canvas, player_board, True)
    draw_board(computer_canvas, computer_board, False)

    # Start the main loop
    root.mainloop()

# Start the game when the script is run
if __name__ == "__main__":
    main()
