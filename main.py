import chess
import chess.pgn
import chess.engine
import cv2
import numpy as np
import threading
import queue
import time
from PIL import Image
from board_to_fen.predict import get_fen_from_image
import os

# === CONFIGURATION ===
VIDEO_PATH = "video.mp4"
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
OUTPUT_SRT = "eval_bar.srt"
STOCKFISH_DEPTH = 20       # Evaluation depth
NUM_WORKERS = 8            # Number of Stockfish worker threads
FRAME_INTERVAL = 60        # Process one frame per second (assuming 60 FPS)

# === GLOBALS ===
fen_queue = queue.Queue()
eval_results = {}

# === Helper: Format Timecode ===
def format_timecode(seconds):
    mins, secs = divmod(int(seconds), 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"00:{mins:02}:{secs:02},{ms:03}"

# === Helper: Draw Eval Bar (16 characters) ===
def draw_eval_bar(eval_score):
    """
    Returns an ASCII-based evaluation bar 16 characters wide.
    Evaluation scores are linearly mapped from -8 to +8:
      - 0.00 => 8 white blocks, 8 black blocks
      - +4.00 => ~12 white blocks, 4 black blocks
      - +8.00 => 16 white blocks; -8.00 => 16 black blocks
    Uses '█' for the filled (white) portion and '░' for the empty (black) portion.
    """
    BAR_LENGTH = 16
    eval_score = max(min(eval_score, 8), -8)
    ratio = 0.5 + (eval_score / 16.0)
    ratio = max(0, min(1, ratio))
    white_blocks = int(round(ratio * BAR_LENGTH))
    black_blocks = BAR_LENGTH - white_blocks
    return "█" * white_blocks + "░" * black_blocks

# === Helper: Emoji Evaluation Conversion ===
def emoji_eval(eval_score):
    """
    Converts a numeric evaluation (formatted as a fixed-width string with a sign)
    into an emoji representation for each digit.
    Uses the following mapping:
      '0': '0️⃣'
      '1': '1️⃣'
      '2': '2️⃣'
      '3': '3️⃣'
      '4': '4️⃣'
      '5': '5️⃣'
      '6': '6️⃣'
      '7': '7️⃣'
      '8': '8️⃣'
      '9': '9️⃣'
      '.': '.'
      '-': '➖'
      '+': '➕'
    """
    mapping = {
        '0': '0️⃣',
        '1': '1️⃣',
        '2': '2️⃣',
        '3': '3️⃣',
        '4': '4️⃣',
        '5': '5️⃣',
        '6': '6️⃣',
        '7': '7️⃣',
        '8': '8️⃣',
        '9': '9️⃣',
        '.': '.',
        '-': '➖',
        '+': '➕'
    }
    # Format the evaluation with a plus sign always, fixed width of 5.2f
    formatted = f"{eval_score:+5.2f}"
    emoji_str = ""
    for ch in formatted:
        emoji_str += mapping.get(ch, ch)
    return emoji_str

# === Helper: Evaluation Description (Not used now) ===
def eval_description(eval_score):
    """
    Returns a textual evaluation description based on the evaluation score.
    (This function is not used in the current SRT output.)
    """
    if -1.0 <= eval_score <= 1.0:
        desc = "A Partida Está Equilibrada"
    elif eval_score > 1.0 and eval_score < 2.0:
        desc = "Vantagem para o Branco"
    elif eval_score >= 2.0:
        desc = "Vantagem decisiva para o Branco"
    elif eval_score < -1.0 and eval_score > -2.0:
        desc = "Vantagem para o Preto"
    elif eval_score <= -2.0:
        desc = "Vantagem decisiva para o Preto"
    else:
        desc = "A Partida Está Equilibrada"
    
    if eval_score >= 0:
        return f"{desc} [+{eval_score:5.2f}]"
    else:
        return f"{desc} [{eval_score:5.2f}]"

# === Helper: Read All PGN Files in Current Folder ===
def get_all_expected_fens():
    """
    Reads all .pgn files in the current folder and builds a dictionary mapping each 
    cropped FEN (board position only) to a tuple (full_fen, move_number).
    If the same cropped FEN appears in multiple games, the first occurrence is used.
    """
    expected = {}
    for filename in os.listdir("."):
        if filename.lower().endswith(".pgn"):
            with open(filename, "r", encoding="latin-1") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    board = game.board()
                    move_num = 0
                    for move in game.mainline_moves():
                        board.push(move)
                        move_num += 1
                        full_fen = board.fen()
                        cropped_fen = full_fen.split(" ")[0]
                        if cropped_fen not in expected:
                            expected[cropped_fen] = (full_fen, move_num)
    return expected

# === Stockfish Worker Thread ===
def stockfish_worker():
    """
    Worker thread that evaluates FENs using Stockfish.
    """
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        while True:
            fen = fen_queue.get()
            if fen is None:
                break  # Stop signal
            try:
                board = chess.Board(fen)
                info = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
                eval_score = info["score"].white().score(mate_score=10000) / 100
                eval_results[fen] = eval_score
                print(f"Evaluated FEN: {fen} → Score: {eval_score}")
            except Exception as e:
                print(f"Stockfish error for FEN {fen}: {e}")
                eval_results[fen] = None
            fen_queue.task_done()

def evaluate_fens_parallel(full_fen_list):
    """
    Evaluates all FENs in parallel using multiple worker threads.
    """
    global eval_results
    eval_results = {}
    workers = []
    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=stockfish_worker, daemon=True)
        t.start()
        workers.append(t)
    for fen in full_fen_list:
        fen_queue.put(fen)
    fen_queue.join()
    for _ in range(NUM_WORKERS):
        fen_queue.put(None)
    for t in workers:
        t.join()
    return eval_results

# === Chessboard Detection ===
def detect_chessboard(frame):
    """
    Detects and extracts the chessboard from a video frame.
    Uses Canny edge detection and contour analysis to find the largest quadrilateral.
    Returns the cropped FEN (board layout only) as detected by get_fen_from_image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_square = None
    max_area = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                largest_square = approx
    if largest_square is not None:
        x, y, w, h = cv2.boundingRect(largest_square)
        board_img = frame[y:y+h, x:x+w]
        pil_img = Image.fromarray(cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB))
        try:
            detected_fen = get_fen_from_image(pil_img)
            return detected_fen.split(" ")[0]
        except Exception as e:
            print(f"Error in FEN detection: {e}")
            return None
    print("No chessboard detected in this frame.")
    return None

# === Process Video and Write SRT Incrementally ===
def process_video(video_path, expected_fens, evaluated_scores, output_file):
    """
    Processes the video and writes an SRT entry only when a new FEN is detected.
    Each subtitle row covers the period from when that FEN is first detected until
    just before the next FEN change. The previous subtitle ends exactly where the new
    one begins.
    Each subtitle displays the 16-character evaluation bar followed by the numeric 
    evaluation (converted to emoji digits) in square brackets.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing Video: {video_path} ({total_frames} frames, {fps} FPS)")
    
    current_subtitle_fen = None
    current_subtitle_start_sec = None
    current_eval = 0.0
    subtitle_index = 1
    
    srt_file = open(output_file, "w")
    
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % FRAME_INTERVAL == 0:
            current_sec = frame_index // fps
            detected = detect_chessboard(frame)
            if detected:
                print(f"Second {current_sec}: Detected FEN: {detected}")
                if detected != current_subtitle_fen:
                    if current_subtitle_fen is not None and current_subtitle_start_sec is not None:
                        end_sec = current_sec  # End previous subtitle at current_sec
                        start_tc = format_timecode(current_subtitle_start_sec)
                        end_tc = format_timecode(end_sec)
                        bar = draw_eval_bar(current_eval)
                        numeric = f"[{emoji_eval(current_eval)}]"
                        srt_file.write(f"{subtitle_index}\n{start_tc} --> {end_tc}\n{bar} {numeric}\n\n")
                        srt_file.flush()
                        print(f"Subtitle {subtitle_index}: {start_tc} --> {end_tc} | {bar} {numeric}")
                        subtitle_index += 1
                    current_subtitle_fen = detected
                    current_subtitle_start_sec = current_sec
                    if detected in expected_fens:
                        full_fen, move_num = expected_fens[detected]
                        new_eval = evaluated_scores.get(full_fen, 0.0)
                        current_eval = new_eval
                        print(f"New FEN (move {move_num}): {detected} with eval {current_eval:.2f}")
                    else:
                        print(f"New FEN: {detected} not in expected; using current_eval {current_eval:.2f}")
            else:
                print(f"Second {current_sec}: No FEN detected")
        frame_index += 1
    
    last_sec = frame_index // fps
    if current_subtitle_fen is not None and current_subtitle_start_sec is not None:
        start_tc = format_timecode(current_subtitle_start_sec)
        end_tc = format_timecode(last_sec)
        bar = draw_eval_bar(current_eval)
        numeric = f"[{emoji_eval(current_eval)}]"
        srt_file.write(f"{subtitle_index}\n{start_tc} --> {end_tc}\n{bar} {numeric}\n\n")
        srt_file.flush()
        print(f"Subtitle {subtitle_index}: {start_tc} --> {end_tc} | {bar} {numeric}")
    
    srt_file.close()
    cap.release()
    print("Video processing complete.")

def emoji_eval(eval_score):
    """
    Converts a numeric evaluation (formatted with a plus sign for positive values)
    into an emoji string for each digit.
    Mapping:
      '0' -> '0️⃣'
      '1' -> '1️⃣'
      '2' -> '2️⃣'
      '3' -> '3️⃣'
      '4' -> '4️⃣'
      '5' -> '5️⃣'
      '6' -> '6️⃣'
      '7' -> '7️⃣'
      '8' -> '8️⃣'
      '9' -> '9️⃣'
      '.' -> '.'
      '-' -> '➖'
      '+' -> '➕'
    """
    mapping = {
        '0': '0️⃣',
        '1': '1️⃣',
        '2': '2️⃣',
        '3': '3️⃣',
        '4': '4️⃣',
        '5': '5️⃣',
        '6': '6️⃣',
        '7': '7️⃣',
        '8': '8️⃣',
        '9': '9️⃣',
        '.': '.',
        '-': '➖',
        '+': '➕'
    }
    formatted = f"{eval_score:+5.2f}"
    result = ""
    for ch in formatted:
        result += mapping.get(ch, ch)
    return result

def main():
    print("Starting Chess Video Analysis")
    with open(OUTPUT_SRT, "w") as f:
        f.write("")
    expected_fens = get_all_expected_fens()
    print(f"Extracted {len(expected_fens)} expected moves from PGN files.")
    full_fen_list = [full for full, _ in expected_fens.values()]
    evaluated_scores = evaluate_fens_parallel(full_fen_list)
    process_video(VIDEO_PATH, expected_fens, evaluated_scores, OUTPUT_SRT)
    print(f"Subtitles saved: {OUTPUT_SRT}")
    print("Chess Video Analysis Completed.")

if __name__ == "__main__":
    main()
