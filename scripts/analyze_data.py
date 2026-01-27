#!/usr/bin/env python3
"""
Chess Training Data Analyzer
============================

Comprehensive analysis of the training data for Global Chess Challenge 2025.
Generates statistics, distributions, and quality metrics.

Usage:
    python scripts/analyze_data.py
    python scripts/analyze_data.py --train_file path/to/train.jsonl --sample_size 10000
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

# Try to import optional dependencies
try:
    import chess
    HAS_CHESS = True
except ImportError:
    HAS_CHESS = False
    print("Warning: python-chess not installed. Some analysis features disabled.")


def load_jsonl_sample(filepath: str, sample_size: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """Load JSONL file, optionally sampling."""
    samples = []
    total_lines = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Total samples in file: {total_lines:,}")
    
    if sample_size and sample_size < len(samples):
        random.seed(seed)
        samples = random.sample(samples, sample_size)
        print(f"Sampled: {sample_size:,} samples")
    
    return samples


def extract_fen_from_text(text: str) -> Optional[str]:
    """Extract FEN string from the text field."""
    # Look for FEN pattern after "Position (FEN):"
    match = re.search(r'Position \(FEN\):\s*([^\n]+)', text)
    if match:
        return match.group(1).strip()
    return None


def extract_move_from_text(text: str) -> Optional[str]:
    """Extract UCI move from the text field."""
    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbnQRBN]?)</uci_move>', text)
    if match:
        return match.group(1).lower()
    return None


def extract_legal_moves(text: str) -> List[str]:
    """Extract legal moves list from the text."""
    match = re.search(r'Legal moves:\s*([^\n]+)', text)
    if match:
        moves = match.group(1).strip()
        return [m.strip() for m in moves.split(',')]
    return []


def extract_rationale(text: str) -> Optional[str]:
    """Extract rationale from the text."""
    match = re.search(r'<rationale>(.*?)</rationale>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def analyze_fen_phase(fen: str) -> str:
    """Determine game phase from FEN (opening/middlegame/endgame)."""
    if not HAS_CHESS:
        return "unknown"
    
    try:
        board = chess.Board(fen)
        piece_count = len(board.piece_map())
        
        if piece_count >= 28:
            return "opening"
        elif piece_count >= 14:
            return "middlegame"
        else:
            return "endgame"
    except:
        return "invalid"


def analyze_position_features(fen: str) -> Dict:
    """Extract position features from FEN."""
    features = {
        "side_to_move": None,
        "castling_rights": None,
        "piece_count": None,
        "has_queens": False,
        "has_pawns": False,
        "is_check": False,
    }
    
    if not HAS_CHESS:
        return features
    
    try:
        board = chess.Board(fen)
        features["side_to_move"] = "white" if board.turn else "black"
        features["castling_rights"] = board.castling_rights
        features["piece_count"] = len(board.piece_map())
        features["has_queens"] = bool(board.queens)
        features["has_pawns"] = bool(board.pawns)
        features["is_check"] = board.is_check()
    except:
        pass
    
    return features


def analyze_weight_distribution(samples: List[Dict]) -> Dict:
    """Analyze the weight distribution."""
    weights = [s.get("weight", 1.0) for s in samples]
    
    if not weights:
        return {}
    
    weights_sorted = sorted(weights)
    n = len(weights)
    
    return {
        "min": min(weights),
        "max": max(weights),
        "mean": sum(weights) / n,
        "median": weights_sorted[n // 2],
        "p25": weights_sorted[n // 4],
        "p75": weights_sorted[3 * n // 4],
        "unique_values": len(set(weights)),
        "distribution": Counter(weights),
    }


def analyze_rationale_patterns(samples: List[Dict]) -> Dict:
    """Analyze rationale patterns."""
    patterns = {
        "has_win_probability": 0,
        "has_opening_name": 0,
        "has_endgame_tablebase": 0,
        "has_position_analysis": 0,
        "rationale_lengths": [],
    }
    
    for sample in samples:
        text = sample.get("text", "")
        rationale = extract_rationale(text)
        
        if rationale:
            patterns["rationale_lengths"].append(len(rationale))
            
            if "Win %" in rationale or "Win Prob" in rationale or "%" in rationale:
                patterns["has_win_probability"] += 1
            if "Opening:" in rationale:
                patterns["has_opening_name"] += 1
            if "tablebase" in rationale.lower() or "Endgame tablebase" in rationale:
                patterns["has_endgame_tablebase"] += 1
            if "Position analysis" in rationale:
                patterns["has_position_analysis"] += 1
    
    return patterns


def analyze_move_types(samples: List[Dict]) -> Dict:
    """Analyze move types (promotions, castling, captures)."""
    move_types = {
        "promotions": 0,
        "castling": 0,  # e1g1, e1c1, e8g8, e8c8
        "total_moves": 0,
    }
    
    castling_moves = {"e1g1", "e1c1", "e8g8", "e8c8"}
    
    for sample in samples:
        text = sample.get("text", "")
        move = extract_move_from_text(text)
        
        if move:
            move_types["total_moves"] += 1
            
            if len(move) == 5:  # Promotion
                move_types["promotions"] += 1
            
            if move in castling_moves:
                move_types["castling"] += 1
    
    return move_types


def analyze_legal_moves_distribution(samples: List[Dict]) -> Dict:
    """Analyze distribution of legal moves count."""
    legal_moves_counts = []
    
    for sample in samples:
        text = sample.get("text", "")
        legal_moves = extract_legal_moves(text)
        if legal_moves:
            legal_moves_counts.append(len(legal_moves))
    
    if not legal_moves_counts:
        return {}
    
    return {
        "min": min(legal_moves_counts),
        "max": max(legal_moves_counts),
        "mean": sum(legal_moves_counts) / len(legal_moves_counts),
        "median": sorted(legal_moves_counts)[len(legal_moves_counts) // 2],
    }


def analyze_token_lengths(samples: List[Dict]) -> Dict:
    """Estimate token lengths."""
    # Rough estimate: ~4 chars per token
    text_lengths = [len(s.get("text", "")) for s in samples]
    estimated_tokens = [l // 4 for l in text_lengths]
    
    if not estimated_tokens:
        return {}
    
    return {
        "min_chars": min(text_lengths),
        "max_chars": max(text_lengths),
        "mean_chars": sum(text_lengths) / len(text_lengths),
        "estimated_min_tokens": min(estimated_tokens),
        "estimated_max_tokens": max(estimated_tokens),
        "estimated_mean_tokens": sum(estimated_tokens) / len(estimated_tokens),
    }


def validate_samples(samples: List[Dict]) -> Dict:
    """Validate sample quality."""
    issues = {
        "missing_fen": 0,
        "missing_move": 0,
        "missing_legal_moves": 0,
        "missing_rationale": 0,
        "invalid_fen": 0,
        "illegal_move": 0,
        "move_not_in_legal_list": 0,
        "uppercase_promotion": 0,
    }
    
    for sample in samples:
        text = sample.get("text", "")
        
        fen = extract_fen_from_text(text)
        move = extract_move_from_text(text)
        legal_moves = extract_legal_moves(text)
        rationale = extract_rationale(text)
        
        if not fen:
            issues["missing_fen"] += 1
        elif HAS_CHESS:
            try:
                board = chess.Board(fen)
            except:
                issues["invalid_fen"] += 1
        
        if not move:
            issues["missing_move"] += 1
        else:
            # Check for uppercase promotion
            if len(move) == 5 and move[-1].isupper():
                issues["uppercase_promotion"] += 1
            
            # Check if move is in legal moves list
            if legal_moves and move not in legal_moves:
                issues["move_not_in_legal_list"] += 1
            
            # Validate move legality
            if HAS_CHESS and fen:
                try:
                    board = chess.Board(fen)
                    chess_move = chess.Move.from_uci(move)
                    if chess_move not in board.legal_moves:
                        issues["illegal_move"] += 1
                except:
                    pass
        
        if not legal_moves:
            issues["missing_legal_moves"] += 1
        
        if not rationale:
            issues["missing_rationale"] += 1
    
    return issues


def generate_report(
    train_file: str,
    val_file: Optional[str] = None,
    sample_size: int = 10000
) -> str:
    """Generate comprehensive analysis report."""
    
    report = []
    report.append("=" * 70)
    report.append("CHESS TRAINING DATA ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Load training data
    print(f"\nLoading training data from: {train_file}")
    train_samples = load_jsonl_sample(train_file, sample_size)
    
    report.append(f"Training file: {train_file}")
    report.append(f"Analyzed samples: {len(train_samples):,}")
    report.append("")
    
    # Weight Distribution
    report.append("-" * 50)
    report.append("WEIGHT DISTRIBUTION")
    report.append("-" * 50)
    weight_stats = analyze_weight_distribution(train_samples)
    if weight_stats:
        report.append(f"  Min weight: {weight_stats['min']:.2f}")
        report.append(f"  Max weight: {weight_stats['max']:.2f}")
        report.append(f"  Mean weight: {weight_stats['mean']:.2f}")
        report.append(f"  Median weight: {weight_stats['median']:.2f}")
        report.append(f"  P25 weight: {weight_stats['p25']:.2f}")
        report.append(f"  P75 weight: {weight_stats['p75']:.2f}")
        report.append(f"  Unique weight values: {weight_stats['unique_values']}")
        report.append("")
        report.append("  Weight value distribution:")
        for weight, count in sorted(weight_stats['distribution'].items()):
            pct = 100 * count / len(train_samples)
            report.append(f"    {weight:.1f}: {count:,} ({pct:.1f}%)")
    report.append("")
    
    # Game Phase Distribution
    if HAS_CHESS:
        report.append("-" * 50)
        report.append("GAME PHASE DISTRIBUTION")
        report.append("-" * 50)
        phases = Counter()
        for sample in train_samples:
            fen = extract_fen_from_text(sample.get("text", ""))
            if fen:
                phase = analyze_fen_phase(fen)
                phases[phase] += 1
        
        for phase, count in phases.most_common():
            pct = 100 * count / len(train_samples)
            report.append(f"  {phase}: {count:,} ({pct:.1f}%)")
        report.append("")
    
    # Side to Move Distribution
    if HAS_CHESS:
        report.append("-" * 50)
        report.append("SIDE TO MOVE DISTRIBUTION")
        report.append("-" * 50)
        sides = Counter()
        for sample in train_samples:
            fen = extract_fen_from_text(sample.get("text", ""))
            if fen:
                features = analyze_position_features(fen)
                if features["side_to_move"]:
                    sides[features["side_to_move"]] += 1
        
        for side, count in sides.most_common():
            pct = 100 * count / len(train_samples)
            report.append(f"  {side}: {count:,} ({pct:.1f}%)")
        report.append("")
    
    # Rationale Patterns
    report.append("-" * 50)
    report.append("RATIONALE PATTERNS")
    report.append("-" * 50)
    rationale_stats = analyze_rationale_patterns(train_samples)
    total = len(train_samples)
    report.append(f"  Has win probability: {rationale_stats['has_win_probability']:,} ({100*rationale_stats['has_win_probability']/total:.1f}%)")
    report.append(f"  Has opening name: {rationale_stats['has_opening_name']:,} ({100*rationale_stats['has_opening_name']/total:.1f}%)")
    report.append(f"  Has tablebase ref: {rationale_stats['has_endgame_tablebase']:,} ({100*rationale_stats['has_endgame_tablebase']/total:.1f}%)")
    report.append(f"  Has position analysis: {rationale_stats['has_position_analysis']:,} ({100*rationale_stats['has_position_analysis']/total:.1f}%)")
    
    if rationale_stats['rationale_lengths']:
        lengths = rationale_stats['rationale_lengths']
        report.append(f"  Rationale length (chars): min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
    report.append("")
    
    # Move Types
    report.append("-" * 50)
    report.append("MOVE TYPES")
    report.append("-" * 50)
    move_types = analyze_move_types(train_samples)
    if move_types["total_moves"] > 0:
        report.append(f"  Total moves: {move_types['total_moves']:,}")
        report.append(f"  Promotions: {move_types['promotions']:,} ({100*move_types['promotions']/move_types['total_moves']:.2f}%)")
        report.append(f"  Castling: {move_types['castling']:,} ({100*move_types['castling']/move_types['total_moves']:.2f}%)")
    report.append("")
    
    # Legal Moves Distribution
    report.append("-" * 50)
    report.append("LEGAL MOVES COUNT DISTRIBUTION")
    report.append("-" * 50)
    legal_stats = analyze_legal_moves_distribution(train_samples)
    if legal_stats:
        report.append(f"  Min legal moves: {legal_stats['min']}")
        report.append(f"  Max legal moves: {legal_stats['max']}")
        report.append(f"  Mean legal moves: {legal_stats['mean']:.1f}")
        report.append(f"  Median legal moves: {legal_stats['median']}")
    report.append("")
    
    # Token Length Estimates
    report.append("-" * 50)
    report.append("TOKEN LENGTH ESTIMATES")
    report.append("-" * 50)
    token_stats = analyze_token_lengths(train_samples)
    if token_stats:
        report.append(f"  Character length: min={token_stats['min_chars']}, max={token_stats['max_chars']}, avg={token_stats['mean_chars']:.0f}")
        report.append(f"  Est. tokens (~4 chars/token): min={token_stats['estimated_min_tokens']}, max={token_stats['estimated_max_tokens']}, avg={token_stats['estimated_mean_tokens']:.0f}")
    report.append("")
    
    # Validation Issues
    report.append("-" * 50)
    report.append("VALIDATION ISSUES")
    report.append("-" * 50)
    issues = validate_samples(train_samples)
    total = len(train_samples)
    for issue, count in sorted(issues.items()):
        pct = 100 * count / total if total > 0 else 0
        status = "[OK]" if count == 0 else f"[!] {count:,} ({pct:.2f}%)"
        report.append(f"  {issue}: {status}")
    report.append("")
    
    # Validation data (if provided)
    if val_file and Path(val_file).exists():
        report.append("=" * 70)
        report.append("VALIDATION DATA")
        report.append("=" * 70)
        print(f"\nLoading validation data from: {val_file}")
        val_samples = load_jsonl_sample(val_file, min(sample_size, 5000))
        report.append(f"Validation samples: {len(val_samples):,}")
        
        val_issues = validate_samples(val_samples)
        report.append("Validation issues:")
        for issue, count in sorted(val_issues.items()):
            pct = 100 * count / len(val_samples) if val_samples else 0
            status = "[OK]" if count == 0 else f"[!] {count:,} ({pct:.2f}%)"
            report.append(f"  {issue}: {status}")
    
    report.append("")
    report.append("=" * 70)
    report.append("ANALYSIS COMPLETE")
    report.append("=" * 70)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze chess training data")
    parser.add_argument(
        "--train_file",
        type=str,
        default="src/division2/data/train (1).jsonl",
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="src/division2/data/val (1).jsonl",
        help="Path to validation data JSONL"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50000,
        help="Number of samples to analyze (for faster processing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for report (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    train_path = script_dir / args.train_file
    val_path = script_dir / args.val_file if args.val_file else None
    
    # Check if files exist
    if not train_path.exists():
        # Try absolute path
        train_path = Path(args.train_file)
    
    if not train_path.exists():
        print(f"Error: Training file not found: {train_path}")
        sys.exit(1)
    
    if val_path and not val_path.exists():
        val_path = Path(args.val_file) if Path(args.val_file).exists() else None
    
    # Generate report
    report = generate_report(
        str(train_path),
        str(val_path) if val_path else None,
        args.sample_size
    )
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
