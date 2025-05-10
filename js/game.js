document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Setup Screen
    const setupScreen = document.getElementById('setup-screen');
    const player1NameInput = document.getElementById('player1-name');
    const player2NameInput = document.getElementById('player2-name');
    const matchesToWinInput = document.getElementById('matches-to-win');
    const startGameButton = document.getElementById('start-game');
    
    // DOM Elements - Game Screen
    const gameScreen = document.getElementById('game-screen');
    const gameBoard = document.getElementById('game-board');
    const playerTurnDisplay = document.getElementById('player-turn');
    const currentPlayerName = document.getElementById('current-player-name');
    const gameResultDisplay = document.getElementById('game-result');
    const resetButton = document.getElementById('reset-button');
    const moveCounter = document.getElementById('move-counter');
    const player1NameDisplay = document.getElementById('player1-name-display');
    const player2NameDisplay = document.getElementById('player2-name-display');
    const player1WinsDisplay = document.getElementById('player1-wins');
    const player2WinsDisplay = document.getElementById('player2-wins');
    const currentMatchDisplay = document.getElementById('current-match');
    const totalMatchesDisplay = document.getElementById('total-matches');
    
    // DOM Elements - Tournament Result Screen
    const tournamentResultScreen = document.getElementById('tournament-result');
    const tournamentWinnerName = document.getElementById('tournament-winner-name');
    const finalScorePlayer1 = document.getElementById('final-score-player1');
    const finalScorePlayer2 = document.getElementById('final-score-player2');
    const newTournamentButton = document.getElementById('new-tournament');
    
    // Game model constants
    const BOARD_SIZE = 8;
    const CELLS_TO_WIN = 4;
    const PLAYER_X = 'x';
    const PLAYER_O = 'o';
    
    // Game state
    let board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(''));
    let currentPlayer = PLAYER_X;
    let gameActive = false;
    let moveCount = 0;
    
    // Player and Tournament state
    let player1Name = 'Jugador 1';
    let player2Name = 'Jugador 2';
    let matchesToWin = 3;
    let player1Wins = 0;
    let player2Wins = 0;
    let currentMatch = 1;
    let tournamentActive = false;
    
    // Setup game event listeners
    startGameButton.addEventListener('click', startTournament);
    resetButton.addEventListener('click', resetMatch);
    newTournamentButton.addEventListener('click', showSetupScreen);
    
    // Show setup screen initially
    showSetupScreen();
    
    // Function to show setup screen
    function showSetupScreen() {
        setupScreen.classList.remove('hidden');
        gameScreen.classList.add('hidden');
        tournamentResultScreen.classList.add('hidden');
        
        // Reset tournament state
        player1Wins = 0;
        player2Wins = 0;
        currentMatch = 1;
        tournamentActive = false;
    }
    
    // Function to start tournament
    function startTournament() {
        // Get player names and matches to win
        player1Name = player1NameInput.value.trim() || 'Jugador 1';
        player2Name = player2NameInput.value.trim() || 'Jugador 2';
        matchesToWin = parseInt(matchesToWinInput.value) || 3;
        
        // Update displays
        player1NameDisplay.textContent = player1Name;
        player2NameDisplay.textContent = player2Name;
        player1WinsDisplay.textContent = player1Wins;
        player2WinsDisplay.textContent = player2Wins;
        currentMatchDisplay.textContent = currentMatch;
        totalMatchesDisplay.textContent = matchesToWin * 2 - 1; // Max possible matches
        
        // Switch screens
        setupScreen.classList.add('hidden');
        gameScreen.classList.remove('hidden');
        tournamentResultScreen.classList.add('hidden');
        
        // Start first match
        tournamentActive = true;
        startMatch();
    }
    
    // Function to start a new match
    function startMatch() {
        // Reset the board
        board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(''));
        currentPlayer = PLAYER_X;
        moveCount = 0;
        gameActive = true;
        
        // Update UI
        moveCounter.textContent = `Movimientos: ${moveCount}`;
        updatePlayerTurn();
        gameResultDisplay.textContent = '';
        createBoard();
    }
    
    // Reset current match
    function resetMatch() {
        startMatch();
    }
    
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
        moveCount++;
        
        // Update the UI
        updateCellUI(row, col);
        moveCounter.textContent = `Movimientos: ${moveCount}`;
        
        // Check for win or draw
        if (checkWin(row, col)) {
            endMatch(false, moveCount);
        } else if (isBoardFull()) {
            endMatch(true, moveCount);
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
        const playerName = currentPlayer === PLAYER_X ? player1Name : player2Name;
        currentPlayerName.textContent = playerName;
        playerTurnDisplay.innerHTML = `Turno: <span id="current-player-name">${playerName}</span> (${currentPlayer.toUpperCase()})`;
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
    
    // End the match and update tournament status
    function endMatch(isDraw, moves) {
        gameActive = false;
        
        if (isDraw) {
            gameResultDisplay.textContent = '¡Empate!';
            // In case of a draw, start a new match without updating scores
            setTimeout(() => {
                currentMatch++;
                currentMatchDisplay.textContent = currentMatch;
                startMatch();
            }, 2000);
        } else {
            const winnerName = currentPlayer === PLAYER_X ? player1Name : player2Name;
            
            // Update win counts
            if (currentPlayer === PLAYER_X) {
                player1Wins++;
                player1WinsDisplay.textContent = player1Wins;
            } else {
                player2Wins++;
                player2WinsDisplay.textContent = player2Wins;
            }
            
            // Display winner with move count
            gameResultDisplay.textContent = `¡${winnerName} ha ganado en ${moves} movimientos!`;
            
            // Check if tournament is over
            if (player1Wins >= matchesToWin || player2Wins >= matchesToWin) {
                endTournament();
            } else {
                // Start next match after delay
                setTimeout(() => {
                    currentMatch++;
                    currentMatchDisplay.textContent = currentMatch;
                    startMatch();
                }, 2000);
            }
        }
    }
    
    // End the tournament and show final screen
    function endTournament() {
        setTimeout(() => {
            gameScreen.classList.add('hidden');
            tournamentResultScreen.classList.remove('hidden');
            
            // Determine tournament winner
            let winnerName;
            if (player1Wins > player2Wins) {
                winnerName = player1Name;
            } else {
                winnerName = player2Name;
            }
            
            // Update tournament result display
            tournamentWinnerName.textContent = winnerName;
            finalScorePlayer1.textContent = player1Wins;
            finalScorePlayer2.textContent = player2Wins;
            
            // Reset tournament status
            tournamentActive = false;
        }, 2000);
    }
});