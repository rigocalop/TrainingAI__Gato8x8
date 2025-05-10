document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Training Controls
    const startTrainingButton = document.getElementById('start-training');
    const pauseTrainingButton = document.getElementById('pause-training');
    const stopTrainingButton = document.getElementById('stop-training');
    const resetModelButton = document.getElementById('reset-model');
    const exportModelButton = document.getElementById('export-model');
    const importModelButton = document.getElementById('import-model');
    const playAgainstModelButton = document.getElementById('play-against-model');
    
    // DOM Elements - Settings
    const iterationsInput = document.getElementById('iterations');
    const gamesPerIterationInput = document.getElementById('games-per-iteration');
    const opponentTypeSelect = document.getElementById('opponent-type');
    const learningRateInput = document.getElementById('learning-rate');
    
    // DOM Elements - Status and Metrics
    const trainingStatus = document.getElementById('training-status');
    const currentIterationSpan = document.getElementById('current-iteration');
    const totalIterationsSpan = document.getElementById('total-iterations');
    const winRateDisplay = document.getElementById('win-rate');
    const policyLossDisplay = document.getElementById('policy-loss');
    const valueLossDisplay = document.getElementById('value-loss');
    const completedGamesDisplay = document.getElementById('completed-games');
    const trainingLog = document.getElementById('training-log');
    
    // DOM Elements - Agent Stats
    const modelWinsDisplay = document.getElementById('model-wins');
    const modelLossesDisplay = document.getElementById('model-losses');
    const modelDrawsDisplay = document.getElementById('model-draws');
    const opponentWinsDisplay = document.getElementById('opponent-wins');
    const opponentLossesDisplay = document.getElementById('opponent-losses');
    const opponentDrawsDisplay = document.getElementById('opponent-draws');
    
    // DOM Elements - Charts
    const winRateChart = document.getElementById('win-rate-chart');
    const lossChart = document.getElementById('loss-chart');
    
    // DOM Elements - Training Board
    const trainingBoard = document.getElementById('training-board');
    const columnCoordinates = document.getElementById('column-coordinates');
    const rowCoordinates = document.getElementById('row-coordinates');
    
    // DOM Elements - History and Analysis
    const trainingHistoryList = document.getElementById('training-history-list');
    const moveAccuracyDisplay = document.getElementById('move-accuracy');
    const avgMovesDisplay = document.getElementById('avg-moves');
    const bestResultDisplay = document.getElementById('best-result');
    
    // Game model constants
    const BOARD_SIZE = 8;
    const CELLS_TO_WIN = 4;
    const PLAYER_X = 'x';
    const PLAYER_O = 'o';
    
    // Training state
    let trainingActive = false;
    let trainingPaused = false;
    let currentIteration = 0;
    let totalIterations = 20;
    let gamesPerIteration = 10;
    let completedGames = 0;
    let opponentType = 'heuristic';
    let learningRate = 0.01;
    
    // Model state (simplified version of the actual AI model)
    let model = {
        weights: Array(100).fill().map(() => Math.random() * 0.1 - 0.05), // Random small weights
        biases: Array(10).fill().map(() => Math.random() * 0.1 - 0.05),
        policyLoss: 1.0,
        valueLoss: 1.0,
        trainedGames: 0,
        
        // Predict next move (simulated)
        predict: function(board) {
            // Simple logic to simulate prediction
            const validMoves = [];
            for (let r = 0; r < BOARD_SIZE; r++) {
                for (let c = 0; c < BOARD_SIZE; c++) {
                    if (board[r][c] === '') {
                        validMoves.push({ row: r, col: c });
                    }
                }
            }
            
            if (validMoves.length === 0) return null;
            
            // Return a random valid move (we'll improve this later)
            return validMoves[Math.floor(Math.random() * validMoves.length)];
        },
        
        // Train model (simulated)
        train: function(gameHistory) {
            // Simulate training by adjusting some weights and biases
            this.weights = this.weights.map(w => w + (Math.random() * 0.02 - 0.01) * learningRate);
            this.biases = this.biases.map(b => b + (Math.random() * 0.02 - 0.01) * learningRate);
            
            // Simulate decreasing loss over time
            this.policyLoss *= (0.995 - Math.random() * 0.01);
            this.valueLoss *= (0.997 - Math.random() * 0.01);
            
            // Add some randomness to make it look realistic
            if (Math.random() < 0.1) {
                this.policyLoss += Math.random() * 0.05;
            }
            if (Math.random() < 0.1) {
                this.valueLoss += Math.random() * 0.05;
            }
            
            this.trainedGames += gameHistory.length;
            return { policyLoss: this.policyLoss, valueLoss: this.valueLoss };
        },
        
        // Save model (simulated)
        save: function() {
            return {
                weights: [...this.weights],
                biases: [...this.biases],
                trainedGames: this.trainedGames
            };
        },
        
        // Load model (simulated)
        load: function(savedModel) {
            this.weights = [...savedModel.weights];
            this.biases = [...savedModel.biases];
            this.trainedGames = savedModel.trainedGames;
        },
        
        // Reset model
        reset: function() {
            this.weights = Array(100).fill().map(() => Math.random() * 0.1 - 0.05);
            this.biases = Array(10).fill().map(() => Math.random() * 0.1 - 0.05);
            this.policyLoss = 1.0;
            this.valueLoss = 1.0;
            this.trainedGames = 0;
        }
    };
    
    // Statistics
    const stats = {
        winRates: [],
        policyLosses: [],
        valueLosses: [],
        modelWins: 0,
        modelLosses: 0,
        modelDraws: 0,
        opponentWins: 0,
        opponentLosses: 0,
        opponentDraws: 0,
        moveAccuracy: 0,
        avgMoves: 0,
        totalMoves: 0,
        gameHistory: [],
        
        // Reset statistics
        reset: function() {
            this.winRates = [];
            this.policyLosses = [];
            this.valueLosses = [];
            this.modelWins = 0;
            this.modelLosses = 0;
            this.modelDraws = 0;
            this.opponentWins = 0;
            this.opponentLosses = 0;
            this.opponentDraws = 0;
            this.moveAccuracy = 0;
            this.avgMoves = 0;
            this.totalMoves = 0;
            this.gameHistory = [];
            
            // Update displays
            updateStatsDisplays();
        },
        
        // Add win rate for iteration
        addWinRate: function(winRate) {
            this.winRates.push(winRate);
            updateWinRateChart();
        },
        
        // Add losses for iteration
        addLosses: function(policyLoss, valueLoss) {
            this.policyLosses.push(policyLoss);
            this.valueLosses.push(valueLoss);
            updateLossChart();
        },
        
        // Add game result
        addGameResult: function(result, moves) {
            if (result === 'model_win') {
                this.modelWins++;
                this.opponentLosses++;
            } else if (result === 'opponent_win') {
                this.modelLosses++;
                this.opponentWins++;
            } else { // draw
                this.modelDraws++;
                this.opponentDraws++;
            }
            
            this.totalMoves += moves;
            const totalGames = this.modelWins + this.modelLosses + this.modelDraws;
            this.avgMoves = totalGames > 0 ? this.totalMoves / totalGames : 0;
            
            // Update displays
            updateStatsDisplays();
        },
        
        // Add game to history
        addGameToHistory: function(game) {
            this.gameHistory.push(game);
            updateGameHistory();
        }
    };
    
    // Initialize game board UI
    function createBoard() {
        // Clear the board
        trainingBoard.innerHTML = '';
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
                trainingBoard.appendChild(cell);
            }
        }
    }
    
    // Update board UI with current state
    function updateBoardUI(boardState) {
        // Clear all cell markings
        document.querySelectorAll('.cell').forEach(cell => {
            cell.classList.remove(PLAYER_X, PLAYER_O);
        });
        
        // Add markings based on board state
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (boardState[row][col] !== '') {
                    const cell = document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
                    cell.classList.add(boardState[row][col]);
                }
            }
        }
    }
    
    // Convert coordinates to algebraic notation
    function toAlgebraicNotation(row, col) {
        const file = String.fromCharCode(65 + col); // Column (A-H)
        const rank = BOARD_SIZE - row; // Row (1-8, inverted)
        return file + rank;
    }
    
    // Update win rate chart
    function updateWinRateChart() {
        winRateChart.innerHTML = '';
        
        if (stats.winRates.length === 0) return;
        
        const barWidth = Math.max(5, Math.min(20, 100 / stats.winRates.length));
        const barGap = Math.max(1, barWidth * 0.2);
        
        stats.winRates.forEach((rate, index) => {
            const bar = document.createElement('div');
            bar.classList.add('chart-bar');
            
            // Calculate position and dimensions
            const left = (barWidth + barGap) * index;
            const height = Math.round(rate * 180); // Scale to fit 200px height (20px margin)
            
            // Set style
            bar.style.left = `${left}px`;
            bar.style.width = `${barWidth}px`;
            bar.style.height = `${height}px`;
            
            // Set color based on win rate
            if (rate > 0.6) {
                bar.style.backgroundColor = '#27ae60'; // green
            } else if (rate > 0.4) {
                bar.style.backgroundColor = '#f39c12'; // orange
            } else {
                bar.style.backgroundColor = '#e74c3c'; // red
            }
            
            // Add tooltip
            bar.title = `Iteración ${index + 1}: ${Math.round(rate * 100)}%`;
            
            winRateChart.appendChild(bar);
        });
    }
    
    // Update loss chart
    function updateLossChart() {
        lossChart.innerHTML = '';
        
        if (stats.policyLosses.length === 0) return;
        
        // Create container for policy loss line
        const policyLine = document.createElement('svg');
        policyLine.style.width = '100%';
        policyLine.style.height = '100%';
        policyLine.style.position = 'absolute';
        policyLine.style.top = '0';
        policyLine.style.left = '0';
        
        // Create container for value loss line
        const valueLine = document.createElement('svg');
        valueLine.style.width = '100%';
        valueLine.style.height = '100%';
        valueLine.style.position = 'absolute';
        valueLine.style.top = '0';
        valueLine.style.left = '0';
        
        // Calculate max loss for scaling
        const maxPolicyLoss = Math.max(...stats.policyLosses);
        const maxValueLoss = Math.max(...stats.valueLosses);
        const maxLoss = Math.max(maxPolicyLoss, maxValueLoss, 0.1); // Ensure non-zero
        
        // Build polyline points
        let policyPoints = '';
        let valuePoints = '';
        
        const width = lossChart.clientWidth;
        const height = lossChart.clientHeight;
        
        stats.policyLosses.forEach((loss, index) => {
            const x = (width / (stats.policyLosses.length - 1 || 1)) * index;
            const y = height - (loss / maxLoss * height * 0.9); // 10% margin
            policyPoints += `${x},${y} `;
        });
        
        stats.valueLosses.forEach((loss, index) => {
            const x = (width / (stats.valueLosses.length - 1 || 1)) * index;
            const y = height - (loss / maxLoss * height * 0.9); // 10% margin
            valuePoints += `${x},${y} `;
        });
        
        // Create policy loss polyline
        const policyPolyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        policyPolyline.setAttribute('points', policyPoints);
        policyPolyline.style.fill = 'none';
        policyPolyline.style.stroke = '#3498db'; // blue
        policyPolyline.style.strokeWidth = '2';
        
        // Create value loss polyline
        const valuePolyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        valuePolyline.setAttribute('points', valuePoints);
        valuePolyline.style.fill = 'none';
        valuePolyline.style.stroke = '#e74c3c'; // red
        valuePolyline.style.strokeWidth = '2';
        
        // Add polylines to SVGs
        policyLine.appendChild(policyPolyline);
        valueLine.appendChild(valuePolyline);
        
        // Add legend
        const legend = document.createElement('div');
        legend.style.position = 'absolute';
        legend.style.bottom = '5px';
        legend.style.right = '5px';
        legend.style.fontSize = '12px';
        legend.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
        legend.style.padding = '3px 5px';
        legend.style.borderRadius = '3px';
        
        legend.innerHTML = `
            <span style="color: #3498db; margin-right: 10px;">● Política</span>
            <span style="color: #e74c3c;">● Valor</span>
        `;
        
        // Add everything to the chart
        lossChart.appendChild(policyLine);
        lossChart.appendChild(valueLine);
        lossChart.appendChild(legend);
    }
    
    // Update game history display
    function updateGameHistory() {
        trainingHistoryList.innerHTML = '';
        
        if (stats.gameHistory.length === 0) {
            const noMovesMsg = document.createElement('div');
            noMovesMsg.classList.add('no-moves-message');
            noMovesMsg.textContent = 'Aún no hay partidas registradas';
            trainingHistoryList.appendChild(noMovesMsg);
            return;
        }
        
        stats.gameHistory.forEach((game, index) => {
            const gameEntry = document.createElement('div');
            gameEntry.classList.add('match-entry');
            
            // Add class based on result
            if (game.result === 'draw') {
                gameEntry.classList.add('draw');
            } else if (game.result === 'model_win') {
                gameEntry.classList.add('player-x-win');
            } else {
                gameEntry.classList.add('player-o-win');
            }
            
            // Create game info
            let resultText;
            if (game.result === 'draw') {
                resultText = 'Empate';
            } else if (game.result === 'model_win') {
                resultText = 'Modelo ganó';
            } else {
                resultText = 'Oponente ganó';
            }
            
            gameEntry.innerHTML = `
                <div>Partida ${index + 1}: ${resultText}</div>
                <div>Movimientos: ${game.moves}</div>
            `;
            
            trainingHistoryList.appendChild(gameEntry);
        });
    }
    
    // Update statistics displays
    function updateStatsDisplays() {
        // Update win rate
        const totalGames = stats.modelWins + stats.modelLosses + stats.modelDraws;
        const winRate = totalGames > 0 ? (stats.modelWins / totalGames * 100).toFixed(1) : '0.0';
        winRateDisplay.textContent = `${winRate}%`;
        
        // Update loss displays
        policyLossDisplay.textContent = model.policyLoss.toFixed(4);
        valueLossDisplay.textContent = model.valueLoss.toFixed(4);
        
        // Update games count
        completedGamesDisplay.textContent = totalGames;
        
        // Update agent stats
        modelWinsDisplay.textContent = stats.modelWins;
        modelLossesDisplay.textContent = stats.modelLosses;
        modelDrawsDisplay.textContent = stats.modelDraws;
        opponentWinsDisplay.textContent = stats.opponentWins;
        opponentLossesDisplay.textContent = stats.opponentLosses;
        opponentDrawsDisplay.textContent = stats.opponentDraws;
        
        // Update analysis metrics
        moveAccuracyDisplay.textContent = `${(stats.moveAccuracy * 100).toFixed(1)}%`;
        avgMovesDisplay.textContent = stats.avgMoves.toFixed(1);
        
        // Update best result
        if (stats.modelWins > 0) {
            const bestWin = Math.max(...stats.winRates) * 100;
            bestResultDisplay.textContent = `${bestWin.toFixed(1)}% Victorias`;
        } else {
            bestResultDisplay.textContent = '-';
        }
    }
    
    // Add log entry
    function addLogEntry(message, type = 'info') {
        const logEntry = document.createElement('div');
        logEntry.classList.add('log-entry', type);
        logEntry.textContent = message;
        trainingLog.appendChild(logEntry);
        
        // Scroll to bottom
        trainingLog.scrollTop = trainingLog.scrollHeight;
    }
    
    // Simulate playing a game
    function playGame() {
        // Create empty board
        let board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(''));
        let currentPlayer = PLAYER_X; // Model always plays as X in training
        let moveCount = 0;
        let gameOver = false;
        let winner = null;
        let moveHistory = [];
        
        // Function to check for win
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
        
        // Function to check if board is full
        function isBoardFull() {
            return board.every(row => row.every(cell => cell !== ''));
        }
        
        // Play until game is over
        while (!gameOver) {
            let move;
            
            if (currentPlayer === PLAYER_X) {
                // Model's turn
                move = model.predict(board);
            } else {
                // Opponent's turn
                move = simulateOpponentMove(board);
            }
            
            if (!move) {
                // No valid moves
                gameOver = true;
                break;
            }
            
            // Make the move
            board[move.row][move.col] = currentPlayer;
            moveCount++;
            
            // Add to history
            moveHistory.push({
                player: currentPlayer,
                position: { row: move.row, col: move.col },
                algebraic: toAlgebraicNotation(move.row, move.col)
            });
            
            // Check for win
            if (checkWin(move.row, move.col)) {
                gameOver = true;
                winner = currentPlayer;
            } else if (isBoardFull()) {
                gameOver = true;
            } else {
                // Switch player
                currentPlayer = currentPlayer === PLAYER_X ? PLAYER_O : PLAYER_X;
            }
        }
        
        // Determine result
        let result;
        if (winner === PLAYER_X) {
            result = 'model_win';
        } else if (winner === PLAYER_O) {
            result = 'opponent_win';
        } else {
            result = 'draw';
        }
        
        // Return game data
        return {
            result,
            moves: moveCount,
            history: moveHistory,
            finalBoard: board
        };
    }
    
    // Simulate opponent move
    function simulateOpponentMove(board) {
        // Different strategies based on opponent type
        switch (opponentType) {
            case 'random':
                return randomMove(board);
            case 'heuristic':
                return heuristicMove(board);
            case 'mixed':
                // 50% chance of heuristic, 50% chance of random
                return Math.random() < 0.5 ? heuristicMove(board) : randomMove(board);
            default:
                return randomMove(board);
        }
    }
    
    // Random move strategy
    function randomMove(board) {
        const validMoves = [];
        
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (board[r][c] === '') {
                    validMoves.push({ row: r, col: c });
                }
            }
        }
        
        if (validMoves.length === 0) return null;
        
        return validMoves[Math.floor(Math.random() * validMoves.length)];
    }
    
    // Heuristic move strategy
    function heuristicMove(board) {
        // First, look for winning move
        const winningMove = findWinningMove(board, PLAYER_O);
        if (winningMove) return winningMove;
        
        // Second, block opponent's winning move
        const blockingMove = findWinningMove(board, PLAYER_X);
        if (blockingMove) return blockingMove;
        
        // Third, prefer center region
        const centerMoves = [];
        for (let r = 2; r < BOARD_SIZE - 2; r++) {
            for (let c = 2; c < BOARD_SIZE - 2; c++) {
                if (board[r][c] === '') {
                    centerMoves.push({ row: r, col: c });
                }
            }
        }
        
        if (centerMoves.length > 0) {
            return centerMoves[Math.floor(Math.random() * centerMoves.length)];
        }
        
        // Otherwise, move randomly
        return randomMove(board);
    }
    
    // Find winning move for player
    function findWinningMove(board, player) {
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (board[r][c] === '') {
                    // Try this move
                    board[r][c] = player;
                    
                    // Check if it's a winning move
                    const isWin = checkWinningPosition(board, r, c);
                    
                    // Undo move
                    board[r][c] = '';
                    
                    if (isWin) {
                        return { row: r, col: c };
                    }
                }
            }
        }
        
        return null;
    }
    
    // Check if position is a winning position
    function checkWinningPosition(board, row, col) {
        const player = board[row][col];
        
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
                board[r][c] === player
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
                board[r][c] === player
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
    
    // Start training
    function startTraining() {
        if (trainingActive && !trainingPaused) {
            addLogEntry('El entrenamiento ya está en curso.', 'warning');
            return;
        }
        
        // If paused, resume
        if (trainingPaused) {
            trainingPaused = false;
            trainingStatus.textContent = `Entrenamiento en Progreso - Iteración ${currentIteration + 1}/${totalIterations}`;
            trainingStatus.className = 'training-status active';
            addLogEntry('Entrenamiento reanudado.', 'info');
            
            // Continue training
            setTimeout(trainingIteration, 100);
            return;
        }
        
        // Get training parameters
        totalIterations = parseInt(iterationsInput.value) || 20;
        gamesPerIteration = parseInt(gamesPerIterationInput.value) || 10;
        opponentType = opponentTypeSelect.value;
        learningRate = parseFloat(learningRateInput.value) || 0.01;
        
        // Update displays
        totalIterationsSpan.textContent = totalIterations;
        currentIterationSpan.textContent = '1';
        
        // Reset statistics
        currentIteration = 0;
        completedGames = 0;
        stats.reset();
        
        // Update UI
        trainingActive = true;
        trainingPaused = false;
        trainingStatus.className = 'training-status active';
        trainingStatus.textContent = `Entrenamiento en Progreso - Iteración 1/${totalIterations}`;
        
        // Log start
        addLogEntry(`Iniciando entrenamiento con ${totalIterations} iteraciones, ${gamesPerIteration} juegos por iteración.`, 'info');
        addLogEntry(`Oponente: ${opponentType}, Tasa de aprendizaje: ${learningRate}`, 'info');
        
        // Start first iteration
        setTimeout(trainingIteration, 100);
    }
    
    // Training iteration
    function trainingIteration() {
        if (!trainingActive || trainingPaused) return;
        
        currentIteration++;
        currentIterationSpan.textContent = currentIteration;
        
        // Play games for this iteration
        addLogEntry(`Iniciando iteración ${currentIteration}. Jugando ${gamesPerIteration} partidas...`, 'info');
        
        // Simulate games
        let iterationGames = [];
        let modelWins = 0;
        
        for (let i = 0; i < gamesPerIteration; i++) {
            const game = playGame();
            iterationGames.push(game);
            
            // Count wins
            if (game.result === 'model_win') {
                modelWins++;
            }
            
            // Add to statistics
            stats.addGameResult(game.result, game.moves);
            stats.addGameToHistory(game);
            
            // Update game count
            completedGames++;
            completedGamesDisplay.textContent = completedGames;
            
            // Show the last game board
            if (i === gamesPerIteration - 1) {
                updateBoardUI(game.finalBoard);
            }
        }
        
        // Calculate win rate for this iteration
        const iterationWinRate = modelWins / gamesPerIteration;
        stats.addWinRate(iterationWinRate);
        
        // Train model on these games
        addLogEntry(`Entrenando modelo con ${iterationGames.length} partidas...`, 'info');
        const losses = model.train(iterationGames);
        
        // Add losses to stats
        stats.addLosses(losses.policyLoss, losses.valueLoss);
        
        // Update displays
        updateStatsDisplays();
        
        // Log progress
        addLogEntry(`Iteración ${currentIteration} completada. Tasa de victoria: ${(iterationWinRate * 100).toFixed(1)}%`, 'success');
        
        // Check if training is complete
        if (currentIteration >= totalIterations) {
            addLogEntry(`Entrenamiento completado. ${totalIterations} iteraciones ejecutadas.`, 'success');
            trainingActive = false;
            trainingStatus.textContent = 'Entrenamiento Completado';
            trainingStatus.className = 'training-status stopped';
        } else {
            // Schedule next iteration
            setTimeout(trainingIteration, 500);
        }
    }
    
    // Pause training
    function pauseTraining() {
        if (!trainingActive) {
            addLogEntry('No hay entrenamiento activo para pausar.', 'warning');
            return;
        }
        
        if (trainingPaused) {
            addLogEntry('El entrenamiento ya está pausado.', 'warning');
            return;
        }
        
        trainingPaused = true;
        trainingStatus.textContent = `Entrenamiento Pausado - Iteración ${currentIteration}/${totalIterations}`;
        trainingStatus.className = 'training-status paused';
        addLogEntry('Entrenamiento pausado.', 'info');
    }
    
    // Stop training
    function stopTraining() {
        if (!trainingActive) {
            addLogEntry('No hay entrenamiento activo para detener.', 'warning');
            return;
        }
        
        trainingActive = false;
        trainingPaused = false;
        trainingStatus.textContent = 'Entrenamiento Detenido';
        trainingStatus.className = 'training-status stopped';
        addLogEntry('Entrenamiento detenido.', 'info');
    }
    
    // Reset model
    function resetModel() {
        model.reset();
        stats.reset();
        
        // Update displays
        updateStatsDisplays();
        
        // Reset charts
        winRateChart.innerHTML = '';
        lossChart.innerHTML = '';
        
        // Reset training status
        trainingActive = false;
        trainingPaused = false;
        currentIteration = 0;
        completedGames = 0;
        
        // Update UI
        trainingStatus.textContent = 'Modelo Reiniciado';
        trainingStatus.className = 'training-status stopped';
        currentIterationSpan.textContent = '0';
        completedGamesDisplay.textContent = '0';
        
        // Create empty board
        const emptyBoard = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(''));
        updateBoardUI(emptyBoard);
        
        // Update history
        trainingHistoryList.innerHTML = '';
        const noMovesMsg = document.createElement('div');
        noMovesMsg.classList.add('no-moves-message');
        noMovesMsg.textContent = 'Aún no hay partidas registradas';
        trainingHistoryList.appendChild(noMovesMsg);
        
        // Log reset
        addLogEntry('Modelo reiniciado. Todas las estadísticas y el progreso se han borrado.', 'info');
    }
    
    // Export model
    function exportModel() {
        const modelData = model.save();
        const blob = new Blob([JSON.stringify(modelData)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const a = document.createElement('a');
        a.href = url;
        a.download = `gato8x8_model_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
        
        addLogEntry('Modelo exportado exitosamente.', 'success');
    }
    
    // Import model
    function importModel() {
        // Create file input
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.onchange = e => {
            const file = e.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = event => {
                try {
                    const modelData = JSON.parse(event.target.result);
                    model.load(modelData);
                    
                    // Reset statistics
                    stats.reset();
                    
                    // Update displays
                    updateStatsDisplays();
                    
                    // Reset charts
                    winRateChart.innerHTML = '';
                    lossChart.innerHTML = '';
                    
                    // Update UI
                    trainingActive = false;
                    trainingPaused = false;
                    currentIteration = 0;
                    completedGames = 0;
                    
                    trainingStatus.textContent = 'Modelo Importado';
                    trainingStatus.className = 'training-status stopped';
                    currentIterationSpan.textContent = '0';
                    completedGamesDisplay.textContent = '0';
                    
                    // Create empty board
                    const emptyBoard = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(''));
                    updateBoardUI(emptyBoard);
                    
                    // Update history
                    trainingHistoryList.innerHTML = '';
                    const noMovesMsg = document.createElement('div');
                    noMovesMsg.classList.add('no-moves-message');
                    noMovesMsg.textContent = 'Aún no hay partidas registradas';
                    trainingHistoryList.appendChild(noMovesMsg);
                    
                    // Log reset
                    addLogEntry(`Modelo importado exitosamente. Entrenado en ${modelData.trainedGames} partidas.`, 'success');
                } catch (error) {
                    addLogEntry(`Error al importar modelo: ${error.message}`, 'error');
                }
            };
            
            reader.readAsText(file);
        };
        
        input.click();
    }
    
    // Play against model
    function playAgainstModel() {
        // Redirect to the main game page (this would be integrated more deeply in a real implementation)
        addLogEntry('Redirigiendo al juego principal...', 'info');
        window.location.href = 'index.html';
    }
    
    // Set up event listeners
    startTrainingButton.addEventListener('click', startTraining);
    pauseTrainingButton.addEventListener('click', pauseTraining);
    stopTrainingButton.addEventListener('click', stopTraining);
    resetModelButton.addEventListener('click', resetModel);
    exportModelButton.addEventListener('click', exportModel);
    importModelButton.addEventListener('click', importModel);
    playAgainstModelButton.addEventListener('click', playAgainstModel);
    
    // Initialize the UI
    createBoard();
    
    // Welcome log entry
    addLogEntry('Bienvenido al entrenamiento de IA para Gato 8x8.', 'info');
    addLogEntry('Configure los parámetros y haga clic en "Iniciar Entrenamiento" para comenzar.', 'info');
});