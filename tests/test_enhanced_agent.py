"""
Division 1: Integration Tests for the Grandmaster Upgrade
==========================================================
Comprehensive tests for all components of the enhanced chess agent.

Run with: pytest tests/test_enhanced_agent.py -v
"""

import pytest
import time
import chess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from division1 import (
    # Opening Book
    OpeningBook,
    BookMove,
    OpeningPhase,
    KIA_MOVES,
    HIPPO_MOVES,
    # Best-of-N
    BestOfNSelector,
    WinProbExtractor,
    Candidate,
    # Time Manager
    TimeManager,
    TimeConfig,
    TimeMode,
    GamePhase,
    PanicFallback,
    # Enhanced Agent
    EnhancedChessAgent,
    EnhancedAgentResponse,
    # Utils
    format_uci_move_for_output,
)


class TestOpeningBook:
    """Tests for the opening book module."""
    
    def test_kia_first_move(self):
        """KIA should start with e2e4."""
        book = OpeningBook()
        board = chess.Board()
        
        move = book.get_book_move(board, move_number=1)
        
        assert move is not None
        assert move.move == "e2e4"
        assert move.phase == OpeningPhase.BOOK
    
    def test_hippo_first_move(self):
        """Hippo should start with g7g6."""
        book = OpeningBook()
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))  # White plays first
        
        move = book.get_book_move(board, move_number=1)
        
        assert move is not None
        assert move.move == "g7g6"
        assert move.phase == OpeningPhase.BOOK
    
    def test_book_legality_check(self):
        """Book moves should only be returned if legal."""
        book = OpeningBook()
        
        # Create a weird position where e2e4 is not legal
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        
        # e2e4 is not legal (pawn already on e4)
        move = book.get_book_move(board, move_number=1)
        
        # Should return an alternative or None
        if move is not None:
            assert move.move != "e2e4"
            assert move.move in [m.uci() for m in board.legal_moves]
    
    def test_full_opening_sequence(self):
        """Test playing through the full opening book."""
        book = OpeningBook()
        board = chess.Board()
        
        moves_played = []
        for i in range(20):
            move = book.get_book_move(board)
            if move is None:
                break
            
            assert move.move in [m.uci() for m in board.legal_moves], \
                f"Book move {move.move} is illegal in position"
            
            moves_played.append(move.move)
            board.push(chess.Move.from_uci(move.move))
        
        # Should have played at least 10 moves from book
        assert len(moves_played) >= 10
    
    def test_kia_moves_count(self):
        """Verify KIA has expected number of moves."""
        assert len(KIA_MOVES) >= 10
    
    def test_hippo_moves_count(self):
        """Verify Hippo has expected number of moves."""
        assert len(HIPPO_MOVES) >= 10


class TestBestOfN:
    """Tests for Best-of-N selection module."""
    
    def test_win_prob_extraction(self):
        """Test Win Probability extraction from various formats."""
        extractor = WinProbExtractor()
        
        test_cases = [
            ("Win Prob: 0.72", 0.72),
            ("Win Probability: 0.85", 0.85),
            ("P(win) = 0.65", 0.65),
            ("85% chance of winning", 0.85),
            ("Win prob: 1.0", 1.0),
            ("Win prob: 0", 0.0),
        ]
        
        for text, expected in test_cases:
            prob, _ = extractor.extract(text)
            assert abs(prob - expected) < 0.01, f"Failed for: {text}"
    
    def test_win_prob_default(self):
        """Default probability should be 0.5 when not found."""
        extractor = WinProbExtractor()
        prob, method = extractor.extract("This is a great move!")
        
        assert prob == 0.5
        assert "default" in method
    
    def test_best_of_n_max_win_prob(self):
        """Test max_win_prob selection strategy."""
        selector = BestOfNSelector(seed=42)
        
        outputs = [
            "<rationale>Win Prob: 0.72</rationale><uci_move>e2e4</uci_move>",
            "<rationale>Win Prob: 0.85</rationale><uci_move>d2d4</uci_move>",
            "<rationale>Win Prob: 0.65</rationale><uci_move>g1f3</uci_move>",
        ]
        legal_moves = {"e2e4", "d2d4", "g1f3"}
        
        result = selector.select(outputs, legal_moves, strategy="max_win_prob")
        
        assert result.selected.move == "d2d4"  # Highest win prob
        assert abs(result.selected.win_prob - 0.85) < 0.01
        assert result.n_valid == 3
    
    def test_best_of_n_majority_vote(self):
        """Test majority_vote selection strategy."""
        selector = BestOfNSelector(seed=42)
        
        outputs = [
            "<rationale>Win Prob: 0.72</rationale><uci_move>e2e4</uci_move>",
            "<rationale>Win Prob: 0.68</rationale><uci_move>e2e4</uci_move>",
            "<rationale>Win Prob: 0.75</rationale><uci_move>e2e4</uci_move>",
            "<rationale>Win Prob: 0.90</rationale><uci_move>d2d4</uci_move>",
        ]
        legal_moves = {"e2e4", "d2d4"}
        
        result = selector.select(outputs, legal_moves, strategy="majority_vote")
        
        assert result.selected.move == "e2e4"  # 3 votes vs 1
    
    def test_best_of_n_handles_invalid(self):
        """Test that invalid moves are filtered out."""
        selector = BestOfNSelector(seed=42)
        
        outputs = [
            "<rationale>Win Prob: 0.90</rationale><uci_move>e2e5</uci_move>",  # Invalid
            "<rationale>Win Prob: 0.70</rationale><uci_move>e2e4</uci_move>",  # Valid
        ]
        legal_moves = {"e2e4", "d2d4"}
        
        result = selector.select(outputs, legal_moves, strategy="max_win_prob")
        
        assert result.selected.move == "e2e4"
        assert result.n_valid == 1
        assert result.n_total == 2


class TestTimeManager:
    """Tests for time management module."""
    
    def test_time_modes(self):
        """Test time mode transitions."""
        config = TimeConfig(
            total_time=10.0,
            panic_threshold=2.0,
            cautious_threshold=5.0,
        )
        manager = TimeManager(config)
        
        # Test normal mode
        manager.start_move()
        assert manager.get_mode() == TimeMode.NORMAL
        
        # Simulate 6 seconds elapsed
        manager._move_start = time.perf_counter() - 6
        assert manager.get_mode() == TimeMode.CAUTIOUS
        
        # Simulate 9 seconds elapsed
        manager._move_start = time.perf_counter() - 9
        assert manager.get_mode() == TimeMode.PANIC
    
    def test_should_panic(self):
        """Test panic detection."""
        manager = TimeManager()
        manager.start_move()
        
        assert not manager.should_panic()
        
        # Simulate 9 seconds elapsed
        manager._move_start = time.perf_counter() - 9
        assert manager.should_panic()
    
    def test_game_phase_detection(self):
        """Test game phase detection by move number."""
        manager = TimeManager()
        
        assert manager._get_game_phase(1) == GamePhase.OPENING
        assert manager._get_game_phase(10) == GamePhase.OPENING
        assert manager._get_game_phase(11) == GamePhase.MIDDLEGAME
        assert manager._get_game_phase(40) == GamePhase.MIDDLEGAME
        assert manager._get_game_phase(41) == GamePhase.ENDGAME
        assert manager._get_game_phase(80) == GamePhase.ENDGAME
    
    def test_decision_n_samples(self):
        """Test that N samples adapts to time budget."""
        manager = TimeManager()
        manager.start_move()
        
        decision_normal = manager.get_decision(move_number=20)
        assert decision_normal.n_samples >= 8
        
        # Simulate time pressure
        manager._move_start = time.perf_counter() - 7
        decision_cautious = manager.get_decision(move_number=20)
        assert decision_cautious.n_samples < decision_normal.n_samples
    
    def test_panic_fallback(self):
        """Test panic fallback returns valid moves."""
        fallback = PanicFallback(seed=42)
        legal_moves = ["e2e4", "d2d4", "g1f3"]
        
        move = fallback.get_fallback_move(legal_moves)
        assert move in legal_moves


class TestEnhancedAgentOffline:
    """Tests for the enhanced agent without LLM."""
    
    def test_agent_creation(self):
        """Test agent can be created."""
        agent = EnhancedChessAgent(use_best_of_n=False)
        assert agent is not None
    
    def test_book_move_without_llm(self):
        """Test that book moves work without LLM."""
        agent = EnhancedChessAgent(use_best_of_n=False)
        agent.new_game()
        
        board = chess.Board()
        legal_moves = [m.uci() for m in board.legal_moves]
        
        response = agent.choose_move(
            fen=board.fen(),
            legal_moves=legal_moves,
            side_to_move="White",
        )
        
        assert response.success
        assert response.is_book_move
        assert response.move == "e2e4"
        assert response.source == "book"
    
    def test_full_game_without_llm(self):
        """Test playing a full game with book + fallback only."""
        agent = EnhancedChessAgent(use_best_of_n=False)
        agent.new_game()
        
        board = chess.Board()
        
        for i in range(50):
            if board.is_game_over():
                break
            
            legal_moves = [m.uci() for m in board.legal_moves]
            if not legal_moves:
                break
            
            response = agent.choose_move(
                fen=board.fen(),
                legal_moves=legal_moves,
                side_to_move="White" if board.turn else "Black",
            )
            
            assert response.success
            assert response.move in legal_moves
            
            board.push(chess.Move.from_uci(response.move))
        
        # Check statistics
        stats = agent.get_stats()
        assert stats["total_moves"] > 0
        assert stats["book_moves"] + stats["fallback_moves"] + stats["panic_moves"] == stats["total_moves"]
    
    def test_formatted_output(self):
        """Test that formatted output matches expected format."""
        agent = EnhancedChessAgent(use_best_of_n=False)
        agent.new_game()
        
        board = chess.Board()
        legal_moves = [m.uci() for m in board.legal_moves]
        
        response = agent.choose_move(
            fen=board.fen(),
            legal_moves=legal_moves,
            side_to_move="White",
        )
        
        formatted = agent.get_formatted_output(response)
        
        assert "<rationale>" in formatted
        assert "</rationale>" in formatted
        assert "<uci_move>" in formatted
        assert "</uci_move>" in formatted
        assert response.move in formatted
    
    def test_never_returns_invalid_move(self):
        """Critical: Agent should NEVER return an invalid move."""
        agent = EnhancedChessAgent(use_best_of_n=False)
        
        # Test 100 random positions
        for _ in range(100):
            board = chess.Board()
            
            # Make some random moves
            for _ in range(20):
                if board.is_game_over():
                    break
                move = list(board.legal_moves)[0]
                board.push(move)
            
            if board.is_game_over():
                continue
            
            legal_moves = [m.uci() for m in board.legal_moves]
            
            response = agent.choose_move(
                fen=board.fen(),
                legal_moves=legal_moves,
                side_to_move="White" if board.turn else "Black",
            )
            
            assert response.move in legal_moves, \
                f"Invalid move {response.move} not in {legal_moves}"


class TestFormatAlignment:
    """Tests for format alignment between Division 1 and Division 2."""
    
    def test_output_format_rationale_first(self):
        """Rationale should come BEFORE uci_move."""
        output = format_uci_move_for_output("e2e4", "Win Prob: 0.72")
        
        # Check order
        rationale_pos = output.find("<rationale>")
        move_pos = output.find("<uci_move>")
        
        assert rationale_pos < move_pos, "Rationale should come before move"
    
    def test_output_format_lowercase_move(self):
        """Move should be lowercase."""
        output = format_uci_move_for_output("E2E4", "test")
        
        assert "e2e4" in output
        assert "E2E4" not in output
    
    def test_output_format_promotion(self):
        """Promotion should be lowercase."""
        output = format_uci_move_for_output("a7a8Q", "test")
        
        assert "a7a8q" in output  # lowercase promotion


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
