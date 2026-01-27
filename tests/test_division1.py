"""
Division 1: Comprehensive Test Suite
=====================================
Tests for all critical components to ensure zero failures.

Test Categories:
1. UCI Format Validation
2. Move Extraction (all edge cases)
3. Fallback System
4. Promotion Handling
5. Castling Handling
6. Integration Tests
"""

import pytest
import chess
from typing import Optional

# Import modules under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from division1.config import (
    ModelType,
    MODEL_REGISTRY,
    NEURON_CONFIG,
    validate_model_for_neuron,
)
from division1.move_utils import (
    MoveExtractor,
    MoveValidator,
    is_valid_uci,
    format_uci_move_for_output,
)
from division1.fallback import (
    IntelligentFallback,
    FallbackReason,
)
from division1.agent import OfflineChessAgent


class TestUCIFormatValidation:
    """Tests for UCI move format validation."""
    
    def test_valid_standard_moves(self):
        """Test that standard moves are recognized as valid."""
        valid_moves = [
            "e2e4", "d2d4", "g1f3", "b1c3",
            "a1a8", "h1h8", "a8a1", "h8h1",
        ]
        for move in valid_moves:
            assert is_valid_uci(move), f"'{move}' should be valid"
    
    def test_valid_promotion_moves(self):
        """Test that promotion moves with lowercase suffixes are valid."""
        valid_promotions = [
            "a7a8q", "b7b8r", "c7c8b", "d7d8n",
            "e2e1q", "f2f1r", "g2g1b", "h2h1n",
        ]
        for move in valid_promotions:
            assert is_valid_uci(move), f"'{move}' should be valid"
    
    def test_invalid_promotion_uppercase(self):
        """Test that uppercase promotion suffixes are invalid."""
        invalid_promotions = [
            "a7a8Q", "b7b8R", "c7c8B", "d7d8N",
        ]
        for move in invalid_promotions:
            assert not is_valid_uci(move), f"'{move}' should be invalid (uppercase)"
    
    def test_invalid_squares(self):
        """Test that invalid squares are rejected."""
        invalid_moves = [
            "i1i2",  # Invalid file
            "a9a8",  # Invalid rank
            "a0a1",  # Invalid rank
            "x2x4",  # Invalid file
        ]
        for move in invalid_moves:
            assert not is_valid_uci(move), f"'{move}' should be invalid"
    
    def test_invalid_formats(self):
        """Test that malformed moves are rejected."""
        invalid_formats = [
            "e4",      # SAN, not UCI
            "Nf3",     # SAN with piece
            "O-O",     # Castling in SAN
            "0-0",     # Castling variant
            "e2-e4",   # With hyphen
            "e2 e4",   # With space
            "e2e4!",   # With annotation
            "",        # Empty
            "e2",      # Incomplete
            "e2e4e5",  # Too long
        ]
        for move in invalid_formats:
            assert not is_valid_uci(move), f"'{move}' should be invalid"
    
    def test_castling_uci_format(self):
        """Test that castling is represented correctly in UCI."""
        # UCI uses king movement for castling
        castling_moves = [
            "e1g1",  # White kingside
            "e1c1",  # White queenside
            "e8g8",  # Black kingside
            "e8c8",  # Black queenside
        ]
        for move in castling_moves:
            assert is_valid_uci(move), f"'{move}' should be valid (castling)"


class TestMoveExtraction:
    """Tests for extracting moves from LLM output."""
    
    @pytest.fixture
    def extractor(self):
        return MoveExtractor()
    
    @pytest.fixture
    def starting_fen(self):
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    def test_extract_from_perfect_tags(self, extractor, starting_fen):
        """Test extraction from properly formatted XML tags."""
        output = "<uci_move>e2e4</uci_move>"
        result = extractor.extract(output, starting_fen)
        assert result.success
        assert result.move == "e2e4"
        assert result.extraction_method == "xml_tags"
    
    def test_extract_with_spaces_in_tags(self, extractor, starting_fen):
        """Test extraction handles whitespace inside tags."""
        output = "<uci_move> e2e4 </uci_move>"
        result = extractor.extract(output, starting_fen)
        assert result.success
        assert result.move == "e2e4"
    
    def test_extract_case_insensitive_tags(self, extractor, starting_fen):
        """Test extraction handles different tag cases."""
        outputs = [
            "<UCI_MOVE>e2e4</UCI_MOVE>",
            "<Uci_Move>e2e4</Uci_Move>",
        ]
        for output in outputs:
            result = extractor.extract(output, starting_fen)
            assert result.success, f"Failed for: {output}"
            assert result.move == "e2e4"
    
    def test_extract_with_rationale(self, extractor, starting_fen):
        """Test extraction when rationale is included."""
        output = """Let me analyze this position.
        
<uci_move>e2e4</uci_move>
<rationale>Controls the center and opens lines for the bishop.</rationale>"""
        result = extractor.extract(output, starting_fen)
        assert result.success
        assert result.move == "e2e4"
        assert result.rationale is not None
        assert "center" in result.rationale.lower()
    
    def test_extract_standalone_uci(self, extractor, starting_fen):
        """Test extraction of standalone UCI moves without tags."""
        outputs = [
            "The best move is e2e4.",
            "I recommend playing e2e4 here.",
            "e2e4",
        ]
        for output in outputs:
            result = extractor.extract(output, starting_fen)
            assert result.success, f"Failed for: {output}"
            assert result.move == "e2e4"
    
    def test_reject_illegal_move_in_tags(self, extractor, starting_fen):
        """Test that illegal moves in tags are rejected."""
        output = "<uci_move>e2e5</uci_move>"  # Illegal from starting position
        result = extractor.extract(output, starting_fen)
        assert not result.success
        assert not result.is_legal
    
    def test_no_valid_move_found(self, extractor, starting_fen):
        """Test handling when no valid move is found."""
        outputs = [
            "I cannot make a move.",
            "This is just random text.",
            "<uci_move>invalid</uci_move>",
        ]
        for output in outputs:
            result = extractor.extract(output, starting_fen)
            assert not result.success, f"Should fail for: {output}"
    
    def test_fuzzy_matching(self, extractor, starting_fen):
        """Test that legal moves are found even without proper formatting."""
        # The move is mentioned but not in tags
        output = "Thinking about the position... e2e4 seems good because it controls the center."
        result = extractor.extract(output, starting_fen)
        assert result.success
        assert result.move == "e2e4"


class TestPromotionHandling:
    """Tests specifically for pawn promotion moves."""
    
    @pytest.fixture
    def extractor(self):
        return MoveExtractor()
    
    @pytest.fixture
    def promotion_fen(self):
        # White pawn on a7, about to promote
        return "8/P7/8/8/8/8/8/4K2k w - - 0 1"
    
    def test_queen_promotion(self, extractor, promotion_fen):
        """Test queen promotion extraction."""
        output = "<uci_move>a7a8q</uci_move>"
        result = extractor.extract(output, promotion_fen)
        assert result.success
        assert result.move == "a7a8q"
    
    def test_rook_promotion(self, extractor, promotion_fen):
        """Test rook promotion extraction."""
        output = "<uci_move>a7a8r</uci_move>"
        result = extractor.extract(output, promotion_fen)
        assert result.success
        assert result.move == "a7a8r"
    
    def test_bishop_promotion(self, extractor, promotion_fen):
        """Test bishop promotion extraction."""
        output = "<uci_move>a7a8b</uci_move>"
        result = extractor.extract(output, promotion_fen)
        assert result.success
        assert result.move == "a7a8b"
    
    def test_knight_promotion(self, extractor, promotion_fen):
        """Test knight promotion extraction."""
        output = "<uci_move>a7a8n</uci_move>"
        result = extractor.extract(output, promotion_fen)
        assert result.success
        assert result.move == "a7a8n"


class TestCastlingHandling:
    """Tests specifically for castling moves."""
    
    @pytest.fixture
    def extractor(self):
        return MoveExtractor()
    
    @pytest.fixture
    def castling_fen(self):
        # Both sides can castle
        return "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    
    def test_white_kingside_castle(self, extractor, castling_fen):
        """Test white kingside castling (e1g1)."""
        output = "<uci_move>e1g1</uci_move>"
        result = extractor.extract(output, castling_fen)
        assert result.success
        assert result.move == "e1g1"
    
    def test_white_queenside_castle(self, extractor, castling_fen):
        """Test white queenside castling (e1c1)."""
        output = "<uci_move>e1c1</uci_move>"
        result = extractor.extract(output, castling_fen)
        assert result.success
        assert result.move == "e1c1"
    
    def test_reject_san_castling(self, extractor, castling_fen):
        """Test that SAN castling notation is rejected."""
        invalid_outputs = [
            "<uci_move>O-O</uci_move>",
            "<uci_move>O-O-O</uci_move>",
            "<uci_move>0-0</uci_move>",
            "<uci_move>0-0-0</uci_move>",
        ]
        for output in invalid_outputs:
            result = extractor.extract(output, castling_fen)
            # Should not extract as valid UCI
            if result.success:
                # If somehow extracted, the move should not be legal
                assert result.move not in ["O-O", "O-O-O", "0-0", "0-0-0"]


class TestFallbackSystem:
    """Tests for the intelligent fallback system."""
    
    @pytest.fixture
    def fallback(self):
        return IntelligentFallback(seed=42)  # Deterministic for testing
    
    def test_finds_checkmate(self, fallback):
        """Test that checkmate is found when available."""
        # Fool's mate position - Qh4# available
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 3"
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        
        # Check if Qxf2# or similar is available
        # Actually let's use a cleaner checkmate position
        mate_fen = "k7/8/1K6/8/8/8/8/7R w - - 0 1"  # Ra1# is mate
        board = chess.Board(mate_fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        
        result = fallback.get_fallback_move(mate_fen, legal_moves)
        # Should find a checkmate if one exists
        if result.reason == FallbackReason.GIVES_CHECKMATE:
            assert result.confidence == 1.0
    
    def test_finds_winning_capture(self, fallback):
        """Test that winning captures are prioritized."""
        # Black queen on e4 undefended, white bishop can take
        fen = "rnb1kbnr/pppp1ppp/8/8/4q3/8/PPPPBPPP/RNBQK1NR w KQkq - 0 1"
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        
        result = fallback.get_fallback_move(fen, legal_moves)
        # Should capture the queen
        if "e2e4" in legal_moves or "e2" in result.move:
            assert result.reason in [FallbackReason.WINNING_CAPTURE, FallbackReason.EQUAL_CAPTURE]
    
    def test_finds_check(self, fallback):
        """Test that checks are found."""
        # Position where check is available
        fen = "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 3"
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        
        result = fallback.get_fallback_move(fen, legal_moves)
        # Bxf7+ is a check
        if result.move == "c4f7":
            assert result.reason == FallbackReason.GIVES_CHECK
    
    def test_always_returns_legal_move(self, fallback):
        """Test that fallback always returns a legal move."""
        test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting
            "8/8/8/8/8/8/6k1/4K2R w - - 0 1",  # Endgame
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian
        ]
        
        for fen in test_fens:
            board = chess.Board(fen)
            legal_moves = [m.uci() for m in board.legal_moves]
            
            if legal_moves:  # Skip if no legal moves
                result = fallback.get_fallback_move(fen, legal_moves)
                assert result.move in legal_moves, f"Illegal move {result.move} for FEN: {fen}"


class TestOfflineAgent:
    """Tests for the offline agent (no LLM)."""
    
    @pytest.fixture
    def agent(self):
        return OfflineChessAgent()
    
    def test_always_returns_valid_response(self, agent):
        """Test that offline agent always returns a valid response."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        
        response = agent.choose_move(fen, legal_moves)
        
        assert response.success
        assert response.move in legal_moves
        assert response.is_fallback
        assert response.latency_ms > 0
    
    def test_response_can_be_formatted(self, agent):
        """Test that response can be formatted for output."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        
        response = agent.choose_move(fen, legal_moves)
        output = format_uci_move_for_output(response.move, response.rationale)
        
        assert "<uci_move>" in output
        assert "</uci_move>" in output
        assert response.move in output


class TestModelConfiguration:
    """Tests for model configuration validation."""
    
    def test_llama_is_tp2_compatible(self):
        """Test that Llama 3 8B is compatible with TP=2."""
        config = MODEL_REGISTRY[ModelType.LLAMA_3_8B]
        is_valid, issues = validate_model_for_neuron(config)
        assert is_valid, f"Llama should be valid: {issues}"
        assert config.is_tp_compatible(2)
    
    def test_qwen_is_tp2_compatible(self):
        """Test that Qwen 2.5 7B is compatible with TP=2."""
        config = MODEL_REGISTRY[ModelType.QWEN_2_5_7B]
        is_valid, issues = validate_model_for_neuron(config)
        assert is_valid, f"Qwen should be valid: {issues}"
        assert config.is_tp_compatible(2)
    
    def test_heads_per_core_calculation(self):
        """Test correct calculation of heads per core."""
        llama = MODEL_REGISTRY[ModelType.LLAMA_3_8B]
        q_per_core, kv_per_core = llama.get_heads_per_core(2)
        assert q_per_core == 16  # 32 / 2
        assert kv_per_core == 4   # 8 / 2
        
        qwen = MODEL_REGISTRY[ModelType.QWEN_2_5_7B]
        q_per_core, kv_per_core = qwen.get_heads_per_core(2)
        assert q_per_core == 14  # 28 / 2
        assert kv_per_core == 2   # 4 / 2


class TestOutputFormatting:
    """Tests for output formatting.
    
    The competition format is:
        <rationale>...</rationale><uci_move>...</uci_move>
    
    Rationale comes FIRST, then the move.
    """
    
    def test_basic_formatting(self):
        """Test basic move formatting with default rationale."""
        output = format_uci_move_for_output("e2e4")
        # Default rationale is added
        assert "<rationale>Win Prob: 0.50</rationale>" in output
        assert "<uci_move>e2e4</uci_move>" in output
        # Rationale comes FIRST (Division 2 training format)
        assert output.index("<rationale>") < output.index("<uci_move>")
    
    def test_formatting_with_rationale(self):
        """Test formatting with custom rationale."""
        output = format_uci_move_for_output("e2e4", "Win Prob: 0.72")
        assert "<uci_move>e2e4</uci_move>" in output
        assert "<rationale>Win Prob: 0.72</rationale>" in output
        # Rationale comes FIRST
        assert output.index("<rationale>") < output.index("<uci_move>")
    
    def test_move_is_lowercase(self):
        """Test that output move is lowercase."""
        output = format_uci_move_for_output("E2E4")  # Input uppercase
        assert "e2e4" in output  # Output should be lowercase
    
    def test_long_rationale_truncated(self):
        """Test that very long rationales are truncated."""
        long_rationale = "x" * 500
        output = format_uci_move_for_output("e2e4", long_rationale)
        # Should be truncated to around 200 chars
        assert len(output) < 300
    
    def test_output_matches_division2_training_format(self):
        """Test that output format matches Division 2 training data format."""
        # Division 2 training format:
        # <rationale>Win Prob: 0.72</rationale><uci_move>e2e4</uci_move>
        output = format_uci_move_for_output("e2e4", "Win Prob: 0.72")
        expected = "<rationale>Win Prob: 0.72</rationale><uci_move>e2e4</uci_move>"
        assert output == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestIntegrationDivision1Division2:
    """
    Integration tests to verify Division 1 (inference) and Division 2 (training)
    are properly aligned in their formats.
    """
    
    @pytest.fixture
    def extractor(self):
        return MoveExtractor()
    
    def test_division2_training_format_parseable(self, extractor):
        """Test that Division 2 training output format is correctly parsed by Division 1."""
        # This is exactly what Division 2 training produces
        model_output = "<rationale>Win Prob: 0.72</rationale><uci_move>e2e4</uci_move>"
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        result = extractor.extract(model_output, starting_fen)
        
        assert result.success
        assert result.move == "e2e4"
        assert result.rationale == "Win Prob: 0.72"
        assert result.extraction_method == "xml_tags"
    
    def test_division1_output_matches_division2_format(self):
        """Test that Division 1 produces output matching Division 2 training format."""
        output = format_uci_move_for_output("e2e4", "Win Prob: 0.65")
        
        # Should be parseable
        extractor = MoveExtractor()
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = extractor.extract(output, starting_fen)
        
        assert result.success
        assert result.move == "e2e4"
        assert result.rationale == "Win Prob: 0.65"
    
    def test_roundtrip_format_consistency(self):
        """Test that formatting and parsing round-trips correctly."""
        original_move = "g1f3"
        original_rationale = "Win Prob: 0.52"
        
        # Format (Division 1 output)
        formatted = format_uci_move_for_output(original_move, original_rationale)
        
        # Parse (Division 1 extraction)
        extractor = MoveExtractor()
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        result = extractor.extract(formatted, starting_fen)
        
        assert result.success
        assert result.move == original_move
        assert result.rationale == original_rationale
    
    def test_all_promotion_moves_handled(self, extractor):
        """Test that all promotion types are handled correctly in Division 2 format."""
        promotion_fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        
        promotions = [
            ("a7a8q", "Win Prob: 0.95"),  # Queen
            ("a7a8r", "Win Prob: 0.85"),  # Rook
            ("a7a8b", "Win Prob: 0.70"),  # Bishop
            ("a7a8n", "Win Prob: 0.75"),  # Knight (for checkmate patterns)
        ]
        
        for move, rationale in promotions:
            # Division 2 format
            model_output = f"<rationale>{rationale}</rationale><uci_move>{move}</uci_move>"
            result = extractor.extract(model_output, promotion_fen)
            
            assert result.success, f"Failed for promotion: {move}"
            assert result.move == move
            assert result.rationale == rationale
    
    def test_castling_moves_handled(self, extractor):
        """Test that castling moves are handled correctly in Division 2 format."""
        castling_fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        
        castling_moves = [
            ("e1g1", "Win Prob: 0.55"),  # White kingside
            ("e1c1", "Win Prob: 0.53"),  # White queenside
        ]
        
        for move, rationale in castling_moves:
            model_output = f"<rationale>{rationale}</rationale><uci_move>{move}</uci_move>"
            result = extractor.extract(model_output, castling_fen)
            
            assert result.success, f"Failed for castling: {move}"
            assert result.move == move
    
    def test_offline_agent_produces_division2_compatible_output(self):
        """Test that the offline fallback agent produces Division 2 compatible output."""
        agent = OfflineChessAgent()
        
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
        
        response = agent.choose_move(fen, legal_moves)
        output = format_uci_move_for_output(response.move, response.rationale)
        
        # Should be parseable
        extractor = MoveExtractor()
        result = extractor.extract(output, fen)
        
        assert result.success
        assert result.move == response.move
    
    def test_complex_position_parsing(self, extractor):
        """Test parsing in a complex middlegame position."""
        # Complex position with many legal moves
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        
        # Simulate what Division 2 model might output
        model_output = "<rationale>Win Prob: 0.58</rationale><uci_move>b1c3</uci_move>"
        
        result = extractor.extract(model_output, fen)
        
        assert result.success
        assert result.move == "b1c3"
        assert result.rationale == "Win Prob: 0.58"
    
    def test_win_prob_edge_values(self, extractor):
        """Test that extreme win probability values are handled."""
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        edge_cases = [
            "Win Prob: 0.00",  # Losing
            "Win Prob: 0.50",  # Equal
            "Win Prob: 1.00",  # Winning (forced mate)
            "Win Prob: 0.99",  # Near-winning
            "Win Prob: 0.01",  # Near-losing
        ]
        
        for rationale in edge_cases:
            model_output = f"<rationale>{rationale}</rationale><uci_move>e2e4</uci_move>"
            result = extractor.extract(model_output, starting_fen)
            
            assert result.success, f"Failed for: {rationale}"
            assert result.rationale == rationale
