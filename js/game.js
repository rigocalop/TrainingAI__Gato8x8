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
    const liveHistoryList = document.getElementById('live-history-list');
    const currentMovesCount = document.getElementById('current-moves-count');
    
    // DOM Elements - Tournament Result Screen
    const tournamentResultScreen = document.getElementById('tournament-result');
    const tournamentWinnerName = document.getElementById('tournament-winner-name');
    const finalScorePlayer1 = document.getElementById('final-score-player1');
    const finalScorePlayer2 = document.getElementById('final-score-player2');
    const matchHistoryList = document.getElementById('match-history-list');
    const totalMovesDisplay = document.getElementById('total-moves');
    const averageMovesDisplay = document.getElementById('average-moves');
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
    let tournamentId = null;
    
    // Match history
    let matchHistory = [];
    let liveHistory = []; // Para el historial en tiempo real
    let playerStats = {}; // Estadísticas de jugadores
    
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
        matchHistory = [];
    }
    
    // Generate a unique ID
    function generateUniqueId() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    }
    
    // Save match history to a file
    function saveMatchHistory(match) {
        const filename = `game_history/match_${match.matchNumber}_${tournamentId}.json`;
        const matchData = JSON.stringify(match, null, 2);
        
        try {
            // En un entorno real, aquí usaríamos fetch o XMLHttpRequest para enviar al servidor
            // Para la demo, simulamos guardarlo localmente (usando localStorage)
            localStorage.setItem(filename, matchData);
            console.log(`Match history saved to ${filename}`);
            return true;
        } catch (error) {
            console.error('Error saving match history:', error);
            return false;
        }
    }
    
    // Save player statistics
    function savePlayerStats() {
        const filename = `player_stats/stats_${tournamentId}.json`;
        const statsData = JSON.stringify(playerStats, null, 2);
        
        try {
            // En un entorno real, aquí usaríamos fetch o XMLHttpRequest para enviar al servidor
            // Para la demo, simulamos guardarlo localmente (usando localStorage)
            localStorage.setItem(filename, statsData);
            console.log(`Player stats saved to ${filename}`);
            return true;
        } catch (error) {
            console.error('Error saving player stats:', error);
            return false;
        }
    }
    
    // Update player statistics
    function updatePlayerStats(match) {
        // Initialize player stats if they don't exist
        if (!playerStats[player1Name]) {
            playerStats[player1Name] = {
                matches: 0,
                wins: 0,
                losses: 0,
                draws: 0,
                totalMoves: 0,
                avgMovesPerWin: 0
            };
        }
        
        if (!playerStats[player2Name]) {
            playerStats[player2Name] = {
                matches: 0,
                wins: 0,
                losses: 0,
                draws: 0,
                totalMoves: 0,
                avgMovesPerWin: 0
            };
        }
        
        // Update statistics based on match result
        if (match.result === 'draw') {
            playerStats[player1Name].matches++;
            playerStats[player1Name].draws++;
            
            playerStats[player2Name].matches++;
            playerStats[player2Name].draws++;
        } else if (match.result === 'player1') {
            playerStats[player1Name].matches++;
            playerStats[player1Name].wins++;
            playerStats[player1Name].totalMoves += match.moves;
            
            playerStats[player2Name].matches++;
            playerStats[player2Name].losses++;
            
            // Update average moves per win
            if (playerStats[player1Name].wins > 0) {
                playerStats[player1Name].avgMovesPerWin = 
                    Math.round((playerStats[player1Name].totalMoves / playerStats[player1Name].wins) * 10) / 10;
            }
        } else { // player2 win
            playerStats[player2Name].matches++;
            playerStats[player2Name].wins++;
            playerStats[player2Name].totalMoves += match.moves;
            
            playerStats[player1Name].matches++;
            playerStats[player1Name].losses++;
            
            // Update average moves per win
            if (playerStats[player2Name].wins > 0) {
                playerStats[player2Name].avgMovesPerWin = 
                    Math.round((playerStats[player2Name].totalMoves / playerStats[player2Name].wins) * 10) / 10;
            }
        }
        
        // Save updated statistics
        savePlayerStats();
    }
    
    // Function to start tournament
    function startTournament() {
        // Generate a unique tournament ID
        tournamentId = generateUniqueId();
        
        // Get player names and matches to win
        player1Name = player1NameInput.value.trim() || 'Jugador 1';
        player2Name = player2NameInput.value.trim() || 'Jugador 2';
        matchesToWin = parseInt(matchesToWinInput.value) || 3;
        
        // Initialize player statistics
        playerStats = {
            [player1Name]: {
                matches: 0,
                wins: 0,
                losses: 0,
                draws: 0,
                totalMoves: 0,
                avgMovesPerWin: 0
            },
            [player2Name]: {
                matches: 0,
                wins: 0,
                losses: 0,
                draws: 0,
                totalMoves: 0,
                avgMovesPerWin: 0
            }
        };
        
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
        liveHistory = []; // Limpiar historial en vivo
        
        // Update UI
        moveCounter.textContent = `Movimientos: ${moveCount}`;
        currentMovesCount.textContent = moveCount;
        updatePlayerTurn();
        gameResultDisplay.textContent = '';
        createBoard();
        updateLiveHistory(); // Actualizar el historial en vivo
    }
    
    // Reset current match
    function resetMatch() {
        startMatch();
    }
    
    // Create the game board UI with chess-like coordinates
    function createBoard() {
        // Clear the board
        gameBoard.innerHTML = '';
        const columnCoordinates = document.getElementById('column-coordinates');
        const rowCoordinates = document.getElementById('row-coordinates');
        columnCoordinates.innerHTML = '';
        rowCoordinates.innerHTML = '';
        
        // Create column coordinates (A-H)
        for (let col = 0; col < BOARD_SIZE; col++) {
            const colCoord = document.createElement('div');
            colCoord.classList.add('board-coordinates', 'column-coordinate');
            colCoord.style.left = `calc(${col} * var(--cell-size) + ${col} * 2px + var(--cell-size) / 2)`;
            colCoord.textContent = String.fromCharCode(65 + col); // A, B, C, etc.
            columnCoordinates.appendChild(colCoord);
        }
        
        // Create row coordinates (1-8)
        for (let row = 0; row < BOARD_SIZE; row++) {
            const rowCoord = document.createElement('div');
            rowCoord.classList.add('board-coordinates', 'row-coordinate');
            rowCoord.style.top = `calc(${row} * var(--cell-size) + ${row} * 2px + var(--cell-size) / 2)`;
            rowCoord.textContent = BOARD_SIZE - row; // 8, 7, 6, etc. (inverted for chess-like coordinates)
            rowCoordinates.appendChild(rowCoord);
        }
        
        // Create the cells
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = row;
                cell.dataset.col = col;
                // Add algebraic notation as data attribute
                cell.dataset.algebraic = getAlgebraicNotation(row, col);
                cell.addEventListener('click', () => handleCellClick(row, col));
                gameBoard.appendChild(cell);
            }
        }
    }
    
    // Convert row, col to algebraic notation (e.g., A8, B3, etc.)
    function getAlgebraicNotation(row, col) {
        const file = String.fromCharCode(65 + col); // Column (A-H)
        const rank = BOARD_SIZE - row; // Row (1-8, inverted)
        return file + rank;
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
        
        // Get algebraic notation for this move
        const algebraicNotation = getAlgebraicNotation(row, col);
        
        // Registrar el movimiento en el historial en vivo
        const moveEntry = {
            player: currentPlayer,
            playerName: currentPlayer === PLAYER_X ? player1Name : player2Name,
            position: { row, col },
            algebraic: algebraicNotation,
            moveNumber: moveCount,
            timestamp: new Date().toISOString()
        };
        liveHistory.push(moveEntry);
        
        // Update the UI
        updateCellUI(row, col);
        moveCounter.textContent = `Movimientos: ${moveCount}`;
        currentMovesCount.textContent = moveCount;
        updateLiveHistory();
        
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
    
    // Update the live history display
    function updateLiveHistory() {
        liveHistoryList.innerHTML = '';
        
        if (liveHistory.length === 0) {
            const noMovesMsg = document.createElement('div');
            noMovesMsg.classList.add('no-moves-message');
            noMovesMsg.textContent = 'Aún no hay movimientos en esta partida';
            liveHistoryList.appendChild(noMovesMsg);
            return;
        }
        
        liveHistory.forEach((move, index) => {
            const moveEntry = document.createElement('div');
            moveEntry.classList.add('match-entry');
            
            // Añadir clase según el jugador
            if (move.player === PLAYER_X) {
                moveEntry.classList.add('player-x-win');
            } else {
                moveEntry.classList.add('player-o-win');
            }
            
            moveEntry.innerHTML = `
                <div>${index + 1}. ${move.playerName} (${move.player.toUpperCase()})</div>
                <div>${move.algebraic}</div>
            `;
            
            liveHistoryList.appendChild(moveEntry);
        });
        
        // Hacer scroll automático al último movimiento
        liveHistoryList.scrollTop = liveHistoryList.scrollHeight;
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
        
        // Add match to history with detailed information
        const matchEntry = {
            matchNumber: currentMatch,
            tournamentId: tournamentId,
            moves: moves,
            result: isDraw ? 'draw' : (currentPlayer === PLAYER_X ? 'player1' : 'player2'),
            player1: player1Name,
            player2: player2Name,
            winner: isDraw ? null : (currentPlayer === PLAYER_X ? player1Name : player2Name),
            moveHistory: [...liveHistory],
            startTime: liveHistory.length > 0 ? liveHistory[0].timestamp : null,
            endTime: new Date().toISOString(),
            boardSize: BOARD_SIZE,
            connectToWin: CELLS_TO_WIN
        };
        
        matchHistory.push(matchEntry);
        
        // Save match history to file
        saveMatchHistory(matchEntry);
        
        // Update player statistics
        updatePlayerStats(matchEntry);
        
        if (isDraw) {
            gameResultDisplay.textContent = '¡Empate!';
            
            // Agregar resultado al historial en vivo
            const drawResult = document.createElement('div');
            drawResult.classList.add('match-entry', 'draw', 'result-entry');
            drawResult.innerHTML = '<div>¡Empate!</div>';
            liveHistoryList.appendChild(drawResult);
            
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
            
            // Agregar resultado al historial en vivo
            const winResult = document.createElement('div');
            winResult.classList.add('match-entry', 'result-entry');
            if (currentPlayer === PLAYER_X) {
                winResult.classList.add('player-x-win');
            } else {
                winResult.classList.add('player-o-win');
            }
            winResult.innerHTML = `<div>¡${winnerName} ha ganado en ${moves} movimientos!</div>`;
            liveHistoryList.appendChild(winResult);
            
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
            
            // Display match history
            displayMatchHistory();
            
            // Save final tournament results
            const tournamentSummary = {
                tournamentId: tournamentId,
                startTime: matchHistory[0].startTime,
                endTime: new Date().toISOString(),
                player1: player1Name,
                player2: player2Name,
                winner: winnerName,
                finalScore: {
                    [player1Name]: player1Wins,
                    [player2Name]: player2Wins
                },
                matchCount: currentMatch,
                matches: matchHistory
            };
            
            // Save tournament summary
            localStorage.setItem(`game_history/tournament_${tournamentId}.json`, JSON.stringify(tournamentSummary, null, 2));
            
            // Reset tournament status
            tournamentActive = false;
        }, 2000);
    }
    
    // Display match history in the result screen
    function displayMatchHistory() {
        matchHistoryList.innerHTML = '';
        
        let totalMoves = 0;
        let completedMatches = 0;
        
        matchHistory.forEach(match => {
            const matchEntryElement = document.createElement('div');
            matchEntryElement.classList.add('match-entry');
            
            // Add class based on result
            if (match.result === 'draw') {
                matchEntryElement.classList.add('draw');
            } else if (match.result === 'player1') {
                matchEntryElement.classList.add('player-x-win');
            } else {
                matchEntryElement.classList.add('player-o-win');
            }
            
            // Match info
            let resultText;
            if (match.result === 'draw') {
                resultText = 'Empate';
            } else if (match.result === 'player1') {
                resultText = `${match.player1} ganó`;
                totalMoves += match.moves;
                completedMatches++;
            } else {
                resultText = `${match.player2} ganó`;
                totalMoves += match.moves;
                completedMatches++;
            }
            
            matchEntryElement.innerHTML = `
                <div>Partida ${match.matchNumber}: ${resultText}</div>
                <div>${match.result !== 'draw' ? match.moves + ' movimientos' : '-'}</div>
            `;
            
            matchHistoryList.appendChild(matchEntryElement);
        });
        
        // Update statistics
        totalMovesDisplay.textContent = totalMoves;
        averageMovesDisplay.textContent = completedMatches > 0
            ? Math.round((totalMoves / completedMatches) * 10) / 10
            : 0;
            
        // Display player statistics
        displayPlayerStats();
    }
    
    // Display player statistics in the tournament result screen
    function displayPlayerStats() {
        // Add a container for player stats if it doesn't exist
        let playerStatsContainer = document.getElementById('player-stats-container');
        if (!playerStatsContainer) {
            playerStatsContainer = document.createElement('div');
            playerStatsContainer.id = 'player-stats-container';
            playerStatsContainer.className = 'match-history';
            playerStatsContainer.innerHTML = '<h3>Estadísticas de Jugadores</h3>';
            
            // Add it after the match history
            const matchHistoryElement = document.querySelector('.match-history');
            matchHistoryElement.parentNode.insertBefore(playerStatsContainer, matchHistoryElement.nextSibling);
        } else {
            playerStatsContainer.innerHTML = '<h3>Estadísticas de Jugadores</h3>';
        }
        
        // Create stats display for each player
        const statsListElement = document.createElement('div');
        statsListElement.className = 'player-stats-list';
        
        // Add player 1 stats
        const player1StatsElement = document.createElement('div');
        player1StatsElement.className = 'player-stats player-x-win';
        player1StatsElement.innerHTML = `
            <h4>${player1Name}</h4>
            <div class="stats-grid">
                <div>Partidas Jugadas: ${playerStats[player1Name].matches}</div>
                <div>Victorias: ${playerStats[player1Name].wins}</div>
                <div>Derrotas: ${playerStats[player1Name].losses}</div>
                <div>Empates: ${playerStats[player1Name].draws}</div>
                <div>Movimientos Totales: ${playerStats[player1Name].totalMoves}</div>
                <div>Prom. Movimientos/Victoria: ${playerStats[player1Name].avgMovesPerWin}</div>
            </div>
        `;
        
        // Add player 2 stats
        const player2StatsElement = document.createElement('div');
        player2StatsElement.className = 'player-stats player-o-win';
        player2StatsElement.innerHTML = `
            <h4>${player2Name}</h4>
            <div class="stats-grid">
                <div>Partidas Jugadas: ${playerStats[player2Name].matches}</div>
                <div>Victorias: ${playerStats[player2Name].wins}</div>
                <div>Derrotas: ${playerStats[player2Name].losses}</div>
                <div>Empates: ${playerStats[player2Name].draws}</div>
                <div>Movimientos Totales: ${playerStats[player2Name].totalMoves}</div>
                <div>Prom. Movimientos/Victoria: ${playerStats[player2Name].avgMovesPerWin}</div>
            </div>
        `;
        
        statsListElement.appendChild(player1StatsElement);
        statsListElement.appendChild(player2StatsElement);
        playerStatsContainer.appendChild(statsListElement);
    }
});