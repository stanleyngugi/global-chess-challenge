"""
Format Compliance Tester

This module tests that the model outputs are correctly formatted
according to competition requirements.

The competition requires:
- UCI move wrapped in <uci_move>...</uci_move> tags
- Move must be legal for the position
- Case sensitivity: promotions must be lowercase (e.g., a7a8q)

This tester helps catch format issues before submission.
"""

import logging
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import chess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test positions covering various scenarios
TEST_POSITIONS = [
    # Starting position
    {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "description": "Starting position",
        "expected_legal": ["e2e4", "d2d4", "g1f3", "b1c3"],
    },
    # Pawn promotion position
    {
        "fen": "8/P7/8/8/8/8/8/4K2k w - - 0 1",
        "description": "Pawn promotion (must be lowercase)",
        "expected_legal": ["a7a8q", "a7a8r", "a7a8b", "a7a8n"],
        "promotion_test": True,
    },
    # Castling position
    {
        "fen": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "description": "Castling (king move, not O-O)",
        "expected_legal": ["e1g1", "e1c1"],  # Castling as king moves
        "castling_test": True,
    },
    # En passant position
    {
        "fen": "8/8/8/3Pp3/8/8/8/4K2k w - e6 0 1",
        "description": "En passant capture",
        "expected_legal": ["d5e6"],  # En passant
        "en_passant_test": True,
    },
    # Complex middlegame
    {
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "description": "Italian Game position",
        "expected_legal": ["O-O", "d2d3", "c2c3"],  # Actually e1g1 in UCI
    },
    # Endgame with few legal moves
    {
        "fen": "8/8/8/8/8/5k2/4p3/4K3 w - - 0 1",
        "description": "Simple endgame",
        "expected_legal": ["e1d1", "e1f1", "e1d2", "e1e2", "e1f2"],
    },
    # Check escape
    {
        "fen": "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1",
        "description": "King in check - must escape",
        "expected_legal": ["e1d1", "e1f1"],
    },
    # Mate in 1
    {
        "fen": "k7/8/1K6/8/8/8/8/7R w - - 0 1",
        "description": "Mate in 1",
        "expected_legal": ["h1a1"],  # Checkmate
    },
]


@dataclass
class FormatTestResult:
    """Result of a format test."""
    position_description: str
    fen: str
    model_output: str
    extracted_move: Optional[str]
    is_valid_format: bool
    is_legal_move: bool
    error_message: Optional[str] = None


class FormatTester:
    """
    Test model output format compliance.
    
    Checks:
    1. Output contains <uci_move>...</uci_move> tags
    2. Move inside tags is valid UCI format
    3. Move is legal for the position
    4. Promotions use lowercase
    5. Castling uses king movement (e1g1, not O-O)
    """
    
    # Regex for valid UCI move
    UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
    
    # Regex for extracting move from tags
    TAG_PATTERN = re.compile(
        r"<uci_move>([a-h][1-8][a-h][1-8][qrbnQRBN]?)</uci_move>",
        re.IGNORECASE
    )
    
    def __init__(self):
        self.results: List[FormatTestResult] = []
    
    def test_output(
        self,
        fen: str,
        model_output: str,
        description: str = ""
    ) -> FormatTestResult:
        """
        Test a single model output for format compliance.
        
        Args:
            fen: Position FEN
            model_output: Raw model output string
            description: Description of the test case
        
        Returns:
            FormatTestResult
        """
        board = chess.Board(fen)
        
        # Try to extract move from tags
        match = self.TAG_PATTERN.search(model_output)
        
        if not match:
            result = FormatTestResult(
                position_description=description,
                fen=fen,
                model_output=model_output[:200],  # Truncate for logging
                extracted_move=None,
                is_valid_format=False,
                is_legal_move=False,
                error_message="No <uci_move> tags found"
            )
            self.results.append(result)
            return result
        
        extracted = match.group(1)
        
        # Check for uppercase promotion (should be lowercase)
        if len(extracted) == 5 and extracted[-1].isupper():
            result = FormatTestResult(
                position_description=description,
                fen=fen,
                model_output=model_output[:200],
                extracted_move=extracted,
                is_valid_format=False,
                is_legal_move=False,
                error_message=f"Promotion must be lowercase: {extracted}"
            )
            self.results.append(result)
            return result
        
        # Normalize to lowercase
        extracted = extracted.lower()
        
        # Check UCI format
        if not self.UCI_PATTERN.match(extracted):
            result = FormatTestResult(
                position_description=description,
                fen=fen,
                model_output=model_output[:200],
                extracted_move=extracted,
                is_valid_format=False,
                is_legal_move=False,
                error_message=f"Invalid UCI format: {extracted}"
            )
            self.results.append(result)
            return result
        
        # Check legality
        try:
            move = chess.Move.from_uci(extracted)
            is_legal = move in board.legal_moves
        except ValueError:
            is_legal = False
        
        if not is_legal:
            result = FormatTestResult(
                position_description=description,
                fen=fen,
                model_output=model_output[:200],
                extracted_move=extracted,
                is_valid_format=True,
                is_legal_move=False,
                error_message=f"Illegal move: {extracted}"
            )
            self.results.append(result)
            return result
        
        # All checks passed
        result = FormatTestResult(
            position_description=description,
            fen=fen,
            model_output=model_output[:200],
            extracted_move=extracted,
            is_valid_format=True,
            is_legal_move=True,
        )
        self.results.append(result)
        return result
    
    def run_test_suite(
        self,
        model_fn: Callable[[str, List[str]], str],
        positions: Optional[List[Dict]] = None,
        num_random_positions: int = 50,
    ) -> Tuple[int, int, Dict]:
        """
        Run a comprehensive format test suite.
        
        Args:
            model_fn: Function that takes (fen, legal_moves) and returns output
            positions: Optional list of test positions
            num_random_positions: Number of random positions to test
        
        Returns:
            Tuple of (passed, failed, detailed_stats)
        """
        positions = positions or TEST_POSITIONS
        
        passed = 0
        failed = 0
        stats = {
            "format_errors": 0,
            "illegal_moves": 0,
            "uppercase_promotions": 0,
            "total_tests": 0,
        }
        
        # Test predefined positions
        for pos in positions:
            fen = pos["fen"]
            desc = pos.get("description", "")
            
            board = chess.Board(fen)
            legal_moves = [m.uci() for m in board.legal_moves]
            
            try:
                output = model_fn(fen, legal_moves)
                result = self.test_output(fen, output, desc)
                
                if result.is_valid_format and result.is_legal_move:
                    passed += 1
                    logger.info(f"PASS: {desc} -> {result.extracted_move}")
                else:
                    failed += 1
                    logger.warning(f"FAIL: {desc} -> {result.error_message}")
                    if not result.is_valid_format:
                        stats["format_errors"] += 1
                        if "uppercase" in (result.error_message or "").lower():
                            stats["uppercase_promotions"] += 1
                    elif not result.is_legal_move:
                        stats["illegal_moves"] += 1
                
            except Exception as e:
                failed += 1
                logger.error(f"ERROR: {desc} -> {e}")
        
        # Test random positions
        random_positions = self._generate_random_positions(num_random_positions)
        for fen, desc in random_positions:
            board = chess.Board(fen)
            legal_moves = [m.uci() for m in board.legal_moves]
            
            try:
                output = model_fn(fen, legal_moves)
                result = self.test_output(fen, output, desc)
                
                if result.is_valid_format and result.is_legal_move:
                    passed += 1
                else:
                    failed += 1
                    if not result.is_valid_format:
                        stats["format_errors"] += 1
                    elif not result.is_legal_move:
                        stats["illegal_moves"] += 1
                        
            except Exception as e:
                failed += 1
        
        stats["total_tests"] = passed + failed
        
        return passed, failed, stats
    
    def _generate_random_positions(
        self,
        count: int
    ) -> List[Tuple[str, str]]:
        """Generate random legal positions for testing."""
        positions = []
        
        for i in range(count):
            board = chess.Board()
            
            # Make random moves
            num_moves = random.randint(5, 60)
            for _ in range(num_moves):
                if board.is_game_over():
                    break
                move = random.choice(list(board.legal_moves))
                board.push(move)
            
            if not board.is_game_over():
                positions.append((board.fen(), f"Random position {i+1}"))
        
        return positions
    
    def get_summary(self) -> Dict:
        """Get summary of all test results."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0}
        
        passed = sum(1 for r in self.results if r.is_valid_format and r.is_legal_move)
        failed = len(self.results) - passed
        
        format_errors = sum(1 for r in self.results if not r.is_valid_format)
        illegal_moves = sum(1 for r in self.results if r.is_valid_format and not r.is_legal_move)
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0,
            "format_errors": format_errors,
            "illegal_moves": illegal_moves,
        }


def test_format_compliance(
    model_path: str,
    num_random: int = 100,
) -> Dict:
    """
    Convenience function to test format compliance.
    
    Args:
        model_path: Path to the model
        num_random: Number of random positions to test
    
    Returns:
        Summary statistics
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
        
        return tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Run tests
    tester = FormatTester()
    passed, failed, stats = tester.run_test_suite(model_fn, num_random_positions=num_random)
    
    summary = tester.get_summary()
    
    print("\n" + "=" * 60)
    print("FORMAT COMPLIANCE TEST RESULTS")
    print("=" * 60)
    print(f"Total tests: {summary['total']}")
    print(f"Passed: {summary['passed']} ({100*summary['pass_rate']:.1f}%)")
    print(f"Failed: {summary['failed']}")
    print(f"  - Format errors: {summary['format_errors']}")
    print(f"  - Illegal moves: {summary['illegal_moves']}")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python format_test.py <model_path> [num_random]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    num_random = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    test_format_compliance(model_path, num_random)
