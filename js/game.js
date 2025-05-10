document.addEventListener('DOMContentLoaded', () => {
    const gameBoard = document.getElementById('game-board');
    const playerTurnDisplay = document.getElementById('player-turn');
    const gameResultDisplay = document.getElementById('game-result');
    const resetButton = document.getElementById('reset-button');
    
    // Game model constants
    const BOARD_SIZE = 8;
    const CELLS_TO_WIN = 4;
    const PLAYER_X = 'x';
    const PLAYER_O = 'o';
    
    // Game state
    let board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(''));
    let currentPlayer = PLAYER_X;
    let gameActive = true;
    
    // Create the game board UI
    function createBoard() {
        gameBoard.innerHTML = '';
        
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.addEventListener('click', () => handleCellClick(row, col));
                gameBoard.appendChild(cell);
            }
        }
    }
    
    // Handle cell click
    function handleCellClick(row, col) {
        // If game is not active or cell is already taken, do nothing
        if (!gameActive || board[row][col] !== '') {
            return;
        }
        
        // Update the board model
        board[row][col] = currentPlayer;
        
        // Update the UI
        updateCellUI(row, col);
        
        // Check for win or draw
        if (checkWin(row, col)) {
            endGame(false);
        } else if (isBoardFull()) {
            endGame(true);
        } else {
            // Switch player
            currentPlayer = currentPlayer === PLAYER_X ? PLAYER_O : PLAYER_X;
            updatePlayerTurn();
        }
    }
    
    // Update cell UI
    function updateCellUI(row, col) {
        const cell = document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
        cell.classList.add(currentPlayer);
    }
    
    // Update the player turn display
    function updatePlayerTurn() {
        playerTurnDisplay.textContent = `Turno: Jugador ${currentPlayer.toUpperCase()}`;
    }
    
    // Check if a player has won
    function checkWin(row, col) {
        // Check in all 8 directions from the last move
        const directions = [
            [0, 1],   // horizontal right
            [1, 0],   // vertical down
            [1, 1],   // diagonal down-right
            [1, -1],  // diagonal down-left
            [0, -1],  // horizontal left
            [-1, 0],  // vertical up
            [-1, -1], // diagonal up-left
            [-1, 1]   // diagonal up-right
        ];
        
        // Check each direction pair (e.g., right and left for horizontal)
        for (let d = 0; d < 4; d++) {
            let count = 1; // Start with 1 for the current cell
            
            // First direction
            let r = row + directions[d][0];
            let c = col + directions[d][1];
            
            while (
                r >= 0 && r < BOARD_SIZE && 
                c >= 0 && c < BOARD_SIZE && 
                board[r][c] === currentPlayer
            ) {
                count++;
                r += directions[d][0];
                c += directions[d][1];
            }
            
            // Opposite direction
            r = row + directions[d + 4][0];
            c = col + directions[d + 4][1];
            
            while (
                r >= 0 && r < BOARD_SIZE && 
                c >= 0 && c < BOARD_SIZE && 
                board[r][c] === currentPlayer
            ) {
                count++;
                r += directions[d + 4][0];
                c += directions[d + 4][1];
            }
            
            // If found 4 or more in a row
            if (count >= CELLS_TO_WIN) {
                return true;
            }
        }
        
        return false;
    }
    
    // Check if the board is full (draw)
    function isBoardFull() {
        return board.every(row => row.every(cell => cell !== ''));
    }
    
    // End the game
    function endGame(isDraw) {
        gameActive = false;
        
        if (isDraw) {
            gameResultDisplay.textContent = '¡Empate!';
        } else {
            gameResultDisplay.textContent = `¡Jugador ${currentPlayer.toUpperCase()} ha ganado!`;
            
            // Highlight the winning cells (would require storing winning cell positions)
        }
    }
    
    // Reset the game
    function resetGame() {
        board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(''));
        currentPlayer = PLAYER_X;
        gameActive = true;
        gameResultDisplay.textContent = '';
        createBoard();
        updatePlayerTurn();
    }
    
    // Event listeners
    resetButton.addEventListener('click', resetGame);
    
    // Initialize the game
    createBoard();
    updatePlayerTurn();
});