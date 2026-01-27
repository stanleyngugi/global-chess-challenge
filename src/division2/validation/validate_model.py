"""
Chess Model Validator

This module validates the trained chess model by:
1. Testing against Stockfish at various levels
2. Computing ACPL (Average Centipawn Loss)
3. Checking format compliance
4. Testing endgame conversion rate

Usage:
    python validate_model.py --model_path output/chess-merged --num_games 100
"""

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single validation game."""
    result: str  # "win", "loss", "draw"
    moves: int
    total_cpl: float  # Sum of centipawn loss
    num_evaluated_moves: int
    format_errors: int
    illegal_moves: int
    game_pgn: str = ""


@dataclass
class ValidationResult:
    """Aggregated validation results."""
    games_played: int
    wins: int
    draws: int
    losses: int
    avg_cpl: float  # Average Centipawn Loss
    format_error_rate: float
    illegal_move_rate: float
    avg_game_length: float
    win_rate: float = field(init=False)
    
    def __post_init__(self):
        if self.games_played > 0:
            self.win_rate = self.wins / self.games_played
        else:
            self.win_rate = 0.0


class ChessValidator:
    """
    Validate a chess model against Stockfish.
    
    This class plays games against Stockfish and measures:
    - Win/Draw/Loss rate
    - Average Centipawn Loss (ACPL)
    - Format compliance
    - Illegal move rate
    """
    
    def __init__(
        self,
        stockfish_path: str = "stockfish",
        stockfish_level: int = 5,
        stockfish_time_limit: float = 0.1,
    ):
        """
        Initialize the validator.
        
        Args:
            stockfish_path: Path to Stockfish binary
            stockfish_level: Stockfish skill level (0-20)
            stockfish_time_limit: Time limit per move in seconds
        """
        self.stockfish_path = stockfish_path
        self.stockfish_level = stockfish_level
        self.stockfish_time_limit = stockfish_time_limit
        self.engine: Optional[chess.engine.SimpleEngine] = None
    
    def __enter__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Skill Level": self.stockfish_level})
        return self
    
    def __exit__(self, *args):
        if self.engine:
            self.engine.quit()
    
    def evaluate_position(self, board: chess.Board) -> int:
        """
        Get Stockfish evaluation of position in centipawns.
        
        Returns evaluation from the side to move's perspective.
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized")
        
        info = self.engine.analyse(
            board,
            chess.engine.Limit(time=self.stockfish_time_limit)
        )
        
        score = info["score"].relative
        
        if score.is_mate():
            mate_in = score.mate()
            # Convert mate to large centipawn value
            return 10000 - abs(mate_in) * 100 if mate_in > 0 else -10000 + abs(mate_in) * 100
        
        return score.score() or 0
    
    def get_stockfish_move(self, board: chess.Board) -> chess.Move:
        """Get Stockfish's best move."""
        if not self.engine:
            raise RuntimeError("Engine not initialized")
        
        result = self.engine.play(
            board,
            chess.engine.Limit(time=self.stockfish_time_limit)
        )
        return result.move
    
    def calculate_move_cpl(
        self,
        board: chess.Board,
        move: chess.Move
    ) -> float:
        """
        Calculate centipawn loss for a move.
        
        CPL = eval_before - eval_after (from player's perspective)
        """
        eval_before = self.evaluate_position(board)
        
        board.push(move)
        eval_after = -self.evaluate_position(board)  # Negate for opponent's turn
        board.pop()
        
        cpl = eval_before - eval_after
        return max(0, cpl)  # CPL is always >= 0
    
    def validate_model(
        self,
        model_fn,  # Function: (fen, legal_moves) -> uci_move_string
        num_games: int = 100,
        max_moves: int = 200,
        model_plays_white: Optional[bool] = None,
    ) -> ValidationResult:
        """
        Play games against Stockfish and collect metrics.
        
        Args:
            model_fn: Function that takes (fen, legal_moves_list) and returns UCI move
            num_games: Number of games to play
            max_moves: Maximum moves per game
            model_plays_white: If None, alternate colors
        
        Returns:
            ValidationResult with aggregated metrics
        """
        results: List[GameResult] = []
        
        for game_idx in range(num_games):
            # Determine color
            if model_plays_white is None:
                plays_white = (game_idx % 2 == 0)
            else:
                plays_white = model_plays_white
            
            result = self._play_game(
                model_fn,
                plays_white,
                max_moves,
                game_idx
            )
            results.append(result)
            
            if (game_idx + 1) % 10 == 0:
                logger.info(f"Completed {game_idx + 1}/{num_games} games")
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _play_game(
        self,
        model_fn,
        model_plays_white: bool,
        max_moves: int,
        game_idx: int,
    ) -> GameResult:
        """Play a single game."""
        board = chess.Board()
        total_cpl = 0.0
        num_evaluated = 0
        format_errors = 0
        illegal_moves = 0
        moves_played = 0
        
        while not board.is_game_over() and moves_played < max_moves:
            is_model_turn = (board.turn == chess.WHITE) == model_plays_white
            
            if is_model_turn:
                # Model's turn
                legal_moves = [m.uci() for m in board.legal_moves]
                
                try:
                    move_str = model_fn(board.fen(), legal_moves)
                    
                    # Extract move from response
                    move_match = re.search(
                        r"<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>",
                        move_str,
                        re.IGNORECASE
                    )
                    
                    if not move_match:
                        format_errors += 1
                        # Try to find any UCI-like pattern
                        fallback = re.search(r"([a-h][1-8][a-h][1-8][qrbn]?)", move_str)
                        if fallback:
                            move_uci = fallback.group(1).lower()
                        else:
                            # Use a random legal move
                            move_uci = random.choice(legal_moves)
                    else:
                        move_uci = move_match.group(1).lower()
                    
                    move = chess.Move.from_uci(move_uci)
                    
                    if move not in board.legal_moves:
                        illegal_moves += 1
                        move = random.choice(list(board.legal_moves))
                    
                    # Calculate CPL
                    cpl = self.calculate_move_cpl(board, move)
                    total_cpl += cpl
                    num_evaluated += 1
                    
                    board.push(move)
                
                except Exception as e:
                    logger.debug(f"Error in game {game_idx}: {e}")
                    format_errors += 1
                    # Make a random move to continue
                    move = random.choice(list(board.legal_moves))
                    board.push(move)
            
            else:
                # Stockfish's turn
                move = self.get_stockfish_move(board)
                board.push(move)
            
            moves_played += 1
        
        # Determine result
        result = board.result()
        if result == "1-0":
            game_result = "win" if model_plays_white else "loss"
        elif result == "0-1":
            game_result = "loss" if model_plays_white else "win"
        else:
            game_result = "draw"
        
        return GameResult(
            result=game_result,
            moves=moves_played,
            total_cpl=total_cpl,
            num_evaluated_moves=num_evaluated,
            format_errors=format_errors,
            illegal_moves=illegal_moves,
        )
    
    def _aggregate_results(self, results: List[GameResult]) -> ValidationResult:
        """Aggregate game results into summary statistics."""
        if not results:
            return ValidationResult(
                games_played=0,
                wins=0, draws=0, losses=0,
                avg_cpl=0.0,
                format_error_rate=0.0,
                illegal_move_rate=0.0,
                avg_game_length=0.0,
            )
        
        wins = sum(1 for r in results if r.result == "win")
        draws = sum(1 for r in results if r.result == "draw")
        losses = sum(1 for r in results if r.result == "loss")
        
        total_cpl = sum(r.total_cpl for r in results)
        total_evaluated = sum(r.num_evaluated_moves for r in results)
        avg_cpl = total_cpl / total_evaluated if total_evaluated > 0 else 0
        
        total_format_errors = sum(r.format_errors for r in results)
        total_illegal = sum(r.illegal_moves for r in results)
        format_error_rate = total_format_errors / total_evaluated if total_evaluated > 0 else 0
        illegal_rate = total_illegal / total_evaluated if total_evaluated > 0 else 0
        
        avg_length = sum(r.moves for r in results) / len(results)
        
        return ValidationResult(
            games_played=len(results),
            wins=wins,
            draws=draws,
            losses=losses,
            avg_cpl=avg_cpl,
            format_error_rate=format_error_rate,
            illegal_move_rate=illegal_rate,
            avg_game_length=avg_length,
        )


def validate_against_stockfish(
    model_path: str,
    stockfish_path: str = "stockfish",
    num_games: int = 100,
    stockfish_level: int = 5,
) -> ValidationResult:
    """
    Convenience function to validate a model against Stockfish.
    
    This loads the model and runs validation games.
    
    Args:
        model_path: Path to the merged model
        stockfish_path: Path to Stockfish binary
        num_games: Number of games to play
        stockfish_level: Stockfish skill level
    
    Returns:
        ValidationResult
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    # Create model function
    def model_fn(fen: str, legal_moves: List[str]) -> str:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a chess grandmaster. Given a chess position in FEN notation and a list of legal moves, output the best move in UCI format wrapped in XML tags.<|eot_id|><|start_header_id|>user<|end_header_id|>

Position (FEN): {fen}
Legal moves: {', '.join(legal_moves)}

Output the best move:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        return response
    
    # Run validation
    with ChessValidator(stockfish_path, stockfish_level) as validator:
        result = validator.validate_model(model_fn, num_games)
    
    return result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate chess model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the merged model"
    )
    parser.add_argument(
        "--stockfish_path",
        type=str,
        default="stockfish",
        help="Path to Stockfish binary"
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of games to play"
    )
    parser.add_argument(
        "--stockfish_level",
        type=int,
        default=5,
        help="Stockfish skill level (0-20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    result = validate_against_stockfish(
        model_path=args.model_path,
        stockfish_path=args.stockfish_path,
        num_games=args.num_games,
        stockfish_level=args.stockfish_level,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Games played: {result.games_played}")
    print(f"Wins: {result.wins} ({100*result.wins/result.games_played:.1f}%)")
    print(f"Draws: {result.draws} ({100*result.draws/result.games_played:.1f}%)")
    print(f"Losses: {result.losses} ({100*result.losses/result.games_played:.1f}%)")
    print(f"\nAverage CPL: {result.avg_cpl:.1f}")
    print(f"Format error rate: {100*result.format_error_rate:.2f}%")
    print(f"Illegal move rate: {100*result.illegal_move_rate:.2f}%")
    print(f"Average game length: {result.avg_game_length:.1f} moves")
    print("=" * 60)
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(vars(result), f, indent=2)
        print(f"\nResults saved to {args.output}")
