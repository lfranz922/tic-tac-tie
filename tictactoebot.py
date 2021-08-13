from __future__ import print_function
import os
import neat
import visualize


class tttBoard():
    """
    -------
    |0|1|2|
    |3|4|5|
    |6|7|8|
    -------
    """

    def __init__(self):
        self.board = [0, 0, 0,
                      0, 0, 0,
                      0, 0, 0]
        self.piece = -1
        self.moves = 0
        self.completed = False

    def cartesianToNum(row, col):
        return (row - 1) + (col - 1) * 3

    """def addPiece(self, row: int, col: int, piece: int = None):
        num = tttBoard.cartesianToNum(row, col)
        if self.board[num] == 0:
            if piece == None:
                self.piece = -self.piece
                piece = self.piece
            self.board[num] = piece

            if self.evaluate(piece) is not None:
                self.completed = True

            return True
        else:
            return False"""

    def addPiece(self, num: int, piece: int = None):
        if self.board[num] == 0:
            if piece == None:
                self.piece = -self.piece
                piece = self.piece
            self.board[num] = piece
            self.moves += 1

            if self.evaluate(piece) is not None or self.moves == 9:
                self.completed = True

            return True
        else:
            return False

    def printBoard(self):
        row_str = ""

        for i in range(len(self.board)):
            spot = self.board[i]
            if spot == 0:
                row_str += " - "
            elif spot == 1:
                row_str += " X "
            elif spot == -1:
                row_str += " O "
            # print(row_str)
            if i % 3 == 2:
                row_str += "\n"
        print(row_str)

    def eval_count(count):
        if count == 3 * piece:
            return True
        elif count == -3 * piece:
            return False
        else:
            return None

    def evaluate(self, piece: int):
        """
        if this piece has won returns True
        if this piece has lost returns False
        otherwise None
        """

        victory = None
        for i in range(3):
            count = self.board[i] + self.board[i + 3] + self.board[i + 6]
            victory = tttBoard.eval_count(count, piece)
            if victory is not None:
                return victory
            count = 0
            for j in range(3):
                count += self.board[i * 3 + j]
            victory = tttBoard.eval_count(count, piece)
            if victory is not None:
                return victory
        count = 0
        for i in range(3):
            count += self.board[4 * i]
        #print("diagonal tl: ", count)
        victory = tttBoard.eval_count(count, piece)
        if victory is not None:
            return victory
        count = 0
        for i in range(3):
            count += self.board[2 * i + 2]
        #print("diagonal bl: ", count)
        victory = tttBoard.eval_count(count, piece)
        if victory is not None:
            return victory
        return victory

    def eval_count(count, piece):
        if count == 3 * piece:
            return True
        elif count == -3 * piece:
            return False
        else:
            return None

def playGame(player1, player2):
    game = tttBoard()
    while not game.completed:

        if game.moves % 2 == 0:
            pieces = player1.activate(tuple(game.board))
        if game.moves % 2 == 1:
            pieces = player2.activate(tuple(game.board))
        piece = pieces.index(max(pieces))

        while not game.addPiece(piece):
            pieces[piece] = -1
            piece = pieces.index(max(pieces))


    game.printBoard()
    evaluation = game.evaluate(1)
    print(evaluation)
    if evaluation == 1:
        return player1
    elif evaluation == -1:
        return player2

def eval_genomes(genomes, config):
    nets = {}
    ids = {}
    for genome_id, genome in genomes:

        genome.fitness = 0
        nets[str(genome_id)] = neat.nn.FeedForwardNetwork.create(genome, config)
        ids[str(nets[str(genome_id)])] = genome_id

    for genome_id1, genome1 in genomes:
        for genome_id2, genome2 in genomes:
            try:
                id = ids[str(playGame(nets[str(genome_id1)], nets[str(genome_id2)]))]
            except:
                id = None
            if id is genome_id1:
                genome1.fitness += 1
            elif id is genome_id2:
                genome2.fitness += 1

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 1)


    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

new = tttBoard()
new.board = [1,0,1,
             1,0,0,
             0,0,0]
"""
new.addPiece(6, piece = 1)
new.printBoard()
#new.addPiece(7)
new.printBoard()
new.addPiece(4, piece = 1)
new.printBoard()
#new.addPiece(1)
new.printBoard()
new.addPiece(5, piece = 1)
"""
new.printBoard()
print(new.evaluate(1))
print(new.board)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
