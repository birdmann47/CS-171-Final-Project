import math
from random import randint
from BoardClasses import Move
from BoardClasses import Board

class StudentAI():

    def __init__(self,col,row,p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col,row,p)
        self.board.initialize_game()
        self.color = ''
        self.opponent = {1:2,2:1}
        self.color = 2

        # MCTS parameters
        self.max_iterations = 200
        self.max_rollout_steps = 50
        self.uct = 1.0


    class MCTSNode:
        def __init__(self, parent, player_to_move, moves, move=None):
            self.parent = parent
            self.player_to_move = player_to_move
            self.children = {}
            self.untried_moves = list(moves)
            self.visits = 0
            self.value = 0.0
            self.move = move

        def is_fully_expanded(self):
            return len(self.untried_moves) == 0

        def has_children(self):
            return len(self.children) > 0

        def select_child_uct(self, C):
            best_score = -1e9
            best_child = None
            for child in self.children.values():
                if child.visits == 0:
                    score = float('inf')
                else:
                    exploitation = child.value / child.visits
                    exploration = C * math.sqrt(math.log(self.visits) / child.visits)
                    score = exploitation + exploration
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child


    def get_move(self, move):

        # Update board with opponent move
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])
        else:
            self.color = 1

        moves_grouped = self.board.get_all_possible_moves(self.color)
        flat_moves = self._flatten_moves(moves_grouped)

        if not flat_moves:
            return Move([])

        if len(flat_moves) == 1:
            chosen_move = flat_moves[0]
            self.board.make_move(chosen_move, self.color)
            return chosen_move

        # Build root node
        root_player = self.color
        root = self.MCTSNode(None, root_player, flat_moves, None)

        # MCTS loop
        for _ in range(self.max_iterations):

            node = root
            current_player = root_player
            last_player = self.opponent[current_player]
            moves_played = []

            # Selection
            while node.is_fully_expanded() and node.has_children():
                child = node.select_child_uct(self.uct)
                self.board.make_move(child.move, current_player)
                moves_played.append(child.move)
                last_player = current_player
                current_player = self.opponent[current_player]
                node = child

            # Terminal node
            result = self._check_terminal(last_player)
            if result is not None:
                self._backpropagate(node, result)
                self._undo_moves(moves_played)
                continue

            # Expansion
            if node.untried_moves:
                idx = randint(0, len(node.untried_moves) - 1)
                move_to_expand = node.untried_moves.pop(idx)

                self.board.make_move(move_to_expand, current_player)
                moves_played.append(move_to_expand)
                last_player = current_player
                current_player = self.opponent[current_player]

                child_moves = self._get_all_moves_flat(current_player)
                child_node = self.MCTSNode(node, current_player, child_moves, move_to_expand)
                node.children[move_to_expand] = child_node
                node = child_node

                # Terminal after expansion
                result = self._check_terminal(last_player)
                if result is not None:
                    self._backpropagate(node, result)
                    self._undo_moves(moves_played)
                    continue

            # Rollout
            result = self._rollout(current_player, last_player, moves_played)

            # Backpropagation
            self._backpropagate(node, result)

            # Reset board
            self._undo_moves(moves_played)

        # Choose best child
        best_child = None
        best_visits = -1
        for child in root.children.values():
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child

        if best_child is None:
            chosen_move = flat_moves[0]
        else:
            chosen_move = best_child.move

        self.board.make_move(chosen_move, self.color)
        return chosen_move


    #HELPERS

    def _flatten_moves(self, moves_grouped):
        flat_moves = []
        for group in moves_grouped:
            for m in group:
                flat_moves.append(m)
        return flat_moves

    def _get_all_moves_flat(self, color_int):
        grouped = self.board.get_all_possible_moves(color_int)
        return self._flatten_moves(grouped)

    def _backpropagate(self, node, result):
        current = node
        while current is not None:
            current.visits += 1
            current.value += result
            current = current.parent

    def _undo_moves(self, moves_played):
        for _ in range(len(moves_played)):
            self.board.undo()

    def _check_terminal(self, last_player):
        status = self.board.is_win(last_player)
        if status == 0:
            return None
        return self._game_result_to_score(status)

    def _rollout(self, current_player, last_player, moves_played):
        steps = 0
        while steps < self.max_rollout_steps:

            status = self.board.is_win(last_player)
            if status != 0:
                return self._game_result_to_score(status)

            moves_grouped = self.board.get_all_possible_moves(current_player)
            flat_moves = self._flatten_moves(moves_grouped)

            if not flat_moves:
                status = self.board.is_win(last_player)
                return self._game_result_to_score(status)

            move = flat_moves[randint(0, len(flat_moves) - 1)]
            self.board.make_move(move, current_player)
            moves_played.append(move)

            last_player = current_player
            current_player = self.opponent[current_player]
            steps += 1

        return 0.5

    def _game_result_to_score(self, status):
        if status == -1:
            return 0.5
        if status == 0:
            return 0.5
        if status == self.color:
            return 1.0
        else:
            return 0.0
