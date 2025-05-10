/**
 * Simplified implementation of the TinyTicTacToeModel for Gato 8x8
 * This is a JavaScript implementation inspired by the Python/PyTorch version
 * but simplified for browser usage.
 */

// Constants
const BOARD_SIZE = 8;
const EMPTY = 0;
const PLAYER_X = 1;
const PLAYER_O = 2;
const WIN_LENGTH = 4;

/**
 * TinyTicTacToeModel - A simplified neural network-like model for Gato 8x8
 */
class TinyTicTacToeModel {
    constructor(config = {}) {
        // Configuration
        this.boardSize = config.boardSize || BOARD_SIZE;
        this.hiddenDim = config.hiddenDim || 64;
        this.temperature = config.temperature || 1.0;
        this.difficulty = config.difficulty || 'medium';
        
        // Initialize weights - In a real neural network these would be trained
        // Here we'll use a simplified approach with heuristics based on difficulty
        this.initialized = true;
    }
    
    /**
     * Convert board to a tensor-like representation
     * @param {Array} board - 2D array representing the board
     * @returns {Array} Flattened array with 3 channels (X, O, empty)
     */
    boardToTensor(board) {
        // Create tensor representation (3 channels: X positions, O positions, empty positions)
        const tensor = new Array(3 * this.boardSize * this.boardSize).fill(0);
        
        // Fill in the tensor
        for (let r = 0; r < this.boardSize; r++) {
            for (let c = 0; c < this.boardSize; c++) {
                const idx = r * this.boardSize + c;
                const piece = board[r][c];
                
                // Channel 0: X positions
                if (piece === PLAYER_X) {
                    tensor[idx] = 1;
                }
                
                // Channel 1: O positions
                if (piece === PLAYER_O) {
                    tensor[idx + this.boardSize * this.boardSize] = 1;
                }
                
                // Channel 2: Empty positions
                if (piece === EMPTY) {
                    tensor[idx + 2 * this.boardSize * this.boardSize] = 1;
                }
            }
        }
        
        return tensor;
    }
    
    /**
     * Create a mask of valid moves
     * @param {Array} board - 2D array representing the board
     * @returns {Array} Boolean array of valid moves
     */
    getValidMovesMask(board) {
        const mask = new Array(this.boardSize * this.boardSize).fill(false);
        
        for (let r = 0; r < this.boardSize; r++) {
            for (let c = 0; c < this.boardSize; c++) {
                const idx = r * this.boardSize + c;
                mask[idx] = (board[r][c] === EMPTY);
            }
        }
        
        return mask;
    }
    
    /**
     * Check if making a move at (row, col) would result in a win
     * @param {Array} board - 2D array representing the board 
     * @param {number} row - Row index
     * @param {number} col - Column index
     * @param {number} player - Player making the move
     * @returns {boolean} True if move would result in win, false otherwise
     */
    checkWin(board, row, col, player) {
        // Clone the board and make the move
        const clonedBoard = board.map(r => [...r]);
        clonedBoard[row][col] = player;
        
        // Check all directions
        const directions = [
            [0, 1],   // horizontal
            [1, 0],   // vertical
            [1, 1],   // diagonal down-right
            [1, -1],  // diagonal down-left
        ];
        
        for (const [dr, dc] of directions) {
            let count = 1;  // Start with 1 for the current piece
            
            // Check in the positive direction
            let r = row + dr;
            let c = col + dc;
            while (r >= 0 && r < this.boardSize && c >= 0 && c < this.boardSize && 
                   clonedBoard[r][c] === player) {
                count++;
                r += dr;
                c += dc;
            }
            
            // Check in the negative direction
            r = row - dr;
            c = col - dc;
            while (r >= 0 && r < this.boardSize && c >= 0 && c < this.boardSize && 
                   clonedBoard[r][c] === player) {
                count++;
                r -= dr;
                c -= dc;
            }
            
            // Check if we have a win
            if (count >= WIN_LENGTH) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * Predict the best move for the current player
     * @param {Array} board - 2D array representing the board
     * @param {number} currentPlayer - The current player (PLAYER_X or PLAYER_O)
     * @returns {Object} The move as {row, col}
     */
    predictMove(board, currentPlayer) {
        // Get valid moves
        const validMovesMask = this.getValidMovesMask(board);
        const validMoves = [];
        
        for (let i = 0; i < validMovesMask.length; i++) {
            if (validMovesMask[i]) {
                const row = Math.floor(i / this.boardSize);
                const col = i % this.boardSize;
                validMoves.push({ row, col });
            }
        }
        
        if (validMoves.length === 0) {
            return null;
        }
        
        // Opponent
        const opponent = currentPlayer === PLAYER_X ? PLAYER_O : PLAYER_X;
        
        // Strategy based on difficulty
        switch (this.difficulty) {
            case 'easy':
                // Random move
                return validMoves[Math.floor(Math.random() * validMoves.length)];
                
            case 'medium':
                // Win if possible or block opponent's win
                
                // First, look for winning move
                for (const move of validMoves) {
                    if (this.checkWin(board, move.row, move.col, currentPlayer)) {
                        return move;
                    }
                }
                
                // Then, look for blocking move
                for (const move of validMoves) {
                    if (this.checkWin(board, move.row, move.col, opponent)) {
                        return move;
                    }
                }
                
                // Otherwise, prefer center region
                const centerMoves = validMoves.filter(
                    move => move.row >= 2 && move.row <= 5 && 
                            move.col >= 2 && move.col <= 5
                );
                
                if (centerMoves.length > 0) {
                    return centerMoves[Math.floor(Math.random() * centerMoves.length)];
                }
                
                // Random move as fallback
                return validMoves[Math.floor(Math.random() * validMoves.length)];
                
            case 'hard':
                // Advanced strategy
                
                // First, look for winning move
                for (const move of validMoves) {
                    if (this.checkWin(board, move.row, move.col, currentPlayer)) {
                        return move;
                    }
                }
                
                // Then, look for blocking move
                for (const move of validMoves) {
                    if (this.checkWin(board, move.row, move.col, opponent)) {
                        return move;
                    }
                }
                
                // Look for two-in-a-row to extend
                const goodMove = this.findGoodMove(board, currentPlayer);
                if (goodMove) {
                    return goodMove;
                }
                
                // Prefer center region
                const strategicMoves = validMoves.filter(
                    move => move.row >= 2 && move.row <= 5 && 
                            move.col >= 2 && move.col <= 5
                );
                
                if (strategicMoves.length > 0) {
                    return strategicMoves[Math.floor(Math.random() * strategicMoves.length)];
                }
                
                // Random move as fallback
                return validMoves[Math.floor(Math.random() * validMoves.length)];
                
            case 'expert':
                // Most advanced strategy - simulates trained model
                
                // First, all the hard mode strategies
                // First, look for winning move
                for (const move of validMoves) {
                    if (this.checkWin(board, move.row, move.col, currentPlayer)) {
                        return move;
                    }
                }
                
                // Then, look for blocking move
                for (const move of validMoves) {
                    if (this.checkWin(board, move.row, move.col, opponent)) {
                        return move;
                    }
                }
                
                // Look for two-in-a-row to extend
                const bestMove = this.findGoodMove(board, currentPlayer);
                if (bestMove) {
                    return bestMove;
                }
                
                // Look for potential trap setup (two potential winning moves)
                const trapMove = this.findTrapMove(board, currentPlayer);
                if (trapMove) {
                    return trapMove;
                }
                
                // Prefer center region
                const centerSquares = validMoves.filter(
                    move => move.row >= 2 && move.row <= 5 && 
                            move.col >= 2 && move.col <= 5
                );
                
                if (centerSquares.length > 0) {
                    return centerSquares[Math.floor(Math.random() * centerSquares.length)];
                }
                
                // Random move as fallback
                return validMoves[Math.floor(Math.random() * validMoves.length)];
                
            default:
                // Default to medium strategy
                return this.predictMove(board, currentPlayer, 'medium');
        }
    }
    
    /**
     * Find a good move (look for a position with potential)
     * @param {Array} board - 2D array representing the board
     * @param {number} player - The current player
     * @returns {Object|null} The good move as {row, col}, or null if none found
     */
    findGoodMove(board, player) {
        // Look for positions that extend our pieces in a row
        for (let r = 0; r < this.boardSize; r++) {
            for (let c = 0; c < this.boardSize; c++) {
                if (board[r][c] === player) {
                    // Check all 8 directions
                    const directions = [
                        [0, 1],   // right
                        [1, 0],   // down
                        [1, 1],   // down-right
                        [1, -1],  // down-left
                        [0, -1],  // left
                        [-1, 0],  // up
                        [-1, -1], // up-left
                        [-1, 1]   // up-right
                    ];
                    
                    for (let d = 0; d < 4; d++) {
                        const [dr, dc] = directions[d];
                        
                        // Check if we have multiple pieces in a row with an empty space after
                        let count = 1;
                        let r2 = r + dr;
                        let c2 = c + dc;
                        
                        // Count pieces in the direction
                        while (r2 >= 0 && r2 < this.boardSize && 
                               c2 >= 0 && c2 < this.boardSize) {
                            if (board[r2][c2] === player) {
                                count++;
                                r2 += dr;
                                c2 += dc;
                            } else if (board[r2][c2] === EMPTY) {
                                // Found an empty space after our pieces
                                if (count >= 2) {
                                    return { row: r2, col: c2 };
                                }
                                break;
                            } else {
                                break;
                            }
                        }
                        
                        // Check the opposite direction
                        r2 = r + directions[d + 4][0];
                        c2 = c + directions[d + 4][1];
                        
                        if (r2 >= 0 && r2 < this.boardSize && 
                            c2 >= 0 && c2 < this.boardSize && 
                            board[r2][c2] === EMPTY && count >= 2) {
                            return { row: r2, col: c2 };
                        }
                    }
                }
            }
        }
        
        return null;
    }
    
    /**
     * Find a trap move (creates two potential winning moves)
     * @param {Array} board - 2D array representing the board
     * @param {number} player - The current player
     * @returns {Object|null} The trap move as {row, col}, or null if none found
     */
    findTrapMove(board, player) {
        const validMoves = [];
        
        // Get all valid moves
        for (let r = 0; r < this.boardSize; r++) {
            for (let c = 0; c < this.boardSize; c++) {
                if (board[r][c] === EMPTY) {
                    validMoves.push({ row: r, col: c });
                }
            }
        }
        
        // For each valid move, check if it creates a trap
        for (const move of validMoves) {
            // Make the move on a cloned board
            const clonedBoard = board.map(r => [...r]);
            clonedBoard[move.row][move.col] = player;
            
            // Count how many winning moves this creates
            let winningMoveCount = 0;
            
            for (let r = 0; r < this.boardSize; r++) {
                for (let c = 0; c < this.boardSize; c++) {
                    if (clonedBoard[r][c] === EMPTY && 
                        this.checkWin(clonedBoard, r, c, player)) {
                        winningMoveCount++;
                        
                        // If we found 2 or more winning moves, this is a trap
                        if (winningMoveCount >= 2) {
                            return move;
                        }
                    }
                }
            }
        }
        
        return null;
    }
    
    /**
     * Load a serialized model (for future use when we have actual trained models)
     * @param {Object} modelData - The serialized model data
     */
    load(modelData) {
        // For now, just copy over configuration
        if (modelData.difficulty) {
            this.difficulty = modelData.difficulty;
        }
        
        if (modelData.temperature) {
            this.temperature = modelData.temperature;
        }
        
        this.initialized = true;
    }
    
    /**
     * Save the model data (for future use)
     * @returns {Object} The serialized model data
     */
    save() {
        return {
            difficulty: this.difficulty,
            temperature: this.temperature
        };
    }
}

// Factory function to create a model with a specific difficulty
function createAIModel(difficulty = 'medium') {
    return new TinyTicTacToeModel({
        difficulty: difficulty,
        temperature: difficulty === 'easy' ? 2.0 : 
                     difficulty === 'medium' ? 1.0 : 0.5
    });
}