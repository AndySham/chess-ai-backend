
from django.core.management.base import BaseCommand
from chess_model.db.models import ChessGame
import chess
import chess.engine
from tqdm import tqdm
import threading

engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")

def iter_games(filename: str):
    pgn = open(filename, "r")
    while True:
        game = chess.pgn.read_game(pgn)
        if game == None:
            break
        yield game

def iter_boards(filename: str):
    for game in iter_games(filename):
        board = game.board()
        yield board
        for move in game.mainline_moves():
            board.push(move)
            board = board.copy(stack=False)
            yield board

def chunk(iter, n):
    curr_chunk = []
    for i in iter:
        if len(curr_chunk) >= n:
            yield curr_chunk
            curr_chunk = []
        curr_chunk.append(i)
    if len(curr_chunk) != 0:
        yield curr_chunk

def write_eval(outio, board: chess.Board, thread_n: int, time_limit: float, mate_score: int):
    epd = board.epd(hmvc=board.halfmove_clock, fmvn=board.fullmove_number)
    eval = engine.analyse(board, 
        chess.engine.Limit(time=time_limit), 
        info=chess.engine.INFO_SCORE,
        options={"Threads": thread_n}
    )["score"].white().score(mate_score=mate_score)
    outio.write("%s,%s\n" % (epd, eval))

class Command(BaseCommand):
    help = 'Create a CSV of evaluations given a PGN file.'

    def add_arguments(self, parser):
        parser.add_argument('--in', type=str, help='Which local file to read PGN data from.')
        parser.add_argument('--out', type=str, help='Where to output evaluations to.')
        parser.add_argument('--max', type=int, default=1_000, help='Maximum number of boards to parse.')
        parser.add_argument('--threads', type=int, default=16, help='Number of threads to calculate with.')
        parser.add_argument('--limit', type=float, default=0.005, help='Length of time to calculate board evaluation.')
        parser.add_argument('--matescore', type=int, default=100_000, help='How much to value a checkmate')

    def handle(self, *args, **options):
        infile = options['in']
        outfile = options['out']
        max_n = options['max']
        thread_n = options['threads']
        time_limit = options['limit']
        mate_score = options['matescore']

        with open(outfile, "w") as out:
            out.write("board,eval\n")
            for i, board in tqdm(enumerate(iter_boards(infile)), total=max_n):
                write_eval(out, board, thread_n, time_limit, mate_score)
                if i >= max_n: 
                    break
   
        