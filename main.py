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
import pytesseract
import csv

# === CONFIGURATION ===
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
OUTPUT_SRT = "eval_bar.srt"
STOCKFISH_DEPTH = 26       # Evaluation depth
NUM_WORKERS = 8            # Number of Stockfish worker threads
FRAME_INTERVAL = 60        # Process one frame per second (assuming 60 FPS)
EVAL_CSV_FILE = "evaluations.csv"

# === GLOBALS ===
fen_queue = queue.Queue()
eval_results = {}

# Standard starting FEN (only board layout)
STANDARD_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

def get_video_files():
    """
    Detects all .mp4 files in the current directory and returns a list of them.
    """
    return [f for f in os.listdir(".") if f.lower().endswith(".mp4")]

def load_evaluations():
    """
    Loads previously computed evaluations from a CSV file.
    Returns a dictionary {FEN: eval_score}.
    """
    evaluations = {}
    if os.path.exists(EVAL_CSV_FILE):
        with open(EVAL_CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    evaluations[row[0]] = float(row[1])  # {FEN: eval_score}
    print(f"Loaded {len(evaluations)} evaluations from CSV.")
    return evaluations

def save_evaluations(evaluations):
    """
    Saves evaluations to a CSV file.
    """
    with open(EVAL_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for fen, score in evaluations.items():
            writer.writerow([fen, score])
    print(f"Saved {len(evaluations)} evaluations to CSV.")

# === Helper: Format Timecode ===
def format_timecode(seconds):
    mins, secs = divmod(int(seconds), 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"00:{mins:02}:{secs:02},{ms:03}"

# === Helper: Draw Eval Bar (24 characters) ===
def draw_eval_bar(eval_score):
    BAR_LENGTH = 24
    if eval_score >= 8:
        ratio = 1.0
    elif eval_score <= -8:
        ratio = 0.0
    else:
        if eval_score >= 4:
            ratio = 0.95
        elif eval_score <= -4:
            ratio = 0.05
        else:
            ratio = 0.05 + ((eval_score + 4) / 8.0) * 0.90
        ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * BAR_LENGTH))
    empty = BAR_LENGTH - filled
    return "▓" * filled + "░" * empty
    
# === Helper: Convert Numeric Evaluation to Fullwidth Symbols ===
def fullwidth_eval(eval_score):
    mapping = {
        '0': '０',
        '1': '１',
        '2': '２',
        '3': '３',
        '4': '４',
        '5': '５',
        '6': '６',
        '7': '７',
        '8': '８',
        '9': '９',
        '.': '.',
        '-': '－',
        '+': '＋'
    }
    formatted = f"{eval_score:+5.2f}"
    return "".join(mapping.get(ch, ch) for ch in formatted)

# === Helper: Read All PGN Files in Current Folder ===
def get_all_expected_fens():
    """
    Reads all .pgn files in the current folder and extracts FENs from both 
    the mainline moves and variations, ensuring legal move application.
    """
    expected = {}

    for filename in os.listdir("."):
        if filename.lower().endswith(".pgn"):
            with open(filename, "r", encoding="latin-1") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break  # No more games in the file

                    stack = [(game, chess.Board(), 0)]  # (Node, Board, Move Number)

                    while stack:
                        node, board, move_num = stack.pop()

                        if node.move is not None:  # If not root node
                            try:
                                board.push(node.move)  # Apply move safely
                                move_num += 1
                                full_fen = board.fen()
                                cropped_fen = full_fen.split(" ")[0]

                                # Store the first occurrence of each FEN
                                if cropped_fen not in expected:
                                    expected[cropped_fen] = (full_fen, move_num)
                            except Exception as e:
                                print(f"Warning: Skipping illegal move {node.move} in {board.fen()}")
                                continue  # Skip bad moves

                        # Push all variations into the stack with a copy of the board
                        for variation in node.variations:
                            stack.append((variation, board.copy(), move_num))

    print(f"Extracted {len(expected)} unique positions from PGN files.")
    return expected

# === Black Orientation Detection ===
def detect_black_perspective(roi):
    """
    Given a candidate board image (roi), crop a small region from its bottom-right,
    upscale, threshold, and run OCR in single-character mode (whitelist a-h).
    Returns True if OCR finds an "a" (indicating Black's perspective), False otherwise.
    """
    h, w = roi.shape[:2]
    crop_size = 30
    y1 = max(0, h - crop_size)
    x1 = max(0, w - crop_size)
    corner = roi[y1:h, x1:w]
    scale_factor = 2.0
    new_w = int(corner.shape[1] * scale_factor)
    new_h = int(corner.shape[0] * scale_factor)
    corner = cv2.resize(corner, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    filtered = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    config_str = r"--psm 10 -c tessedit_char_whitelist=abcdefgh"
    text = pytesseract.image_to_string(filtered, config=config_str).strip().lower()
    print(f"OCR text in corner: '{text}'")
    return 'a' in text

# === Chessboard Detection with New Features ===
def detect_chessboard(frame):
    """
    Detects the largest quadrilateral in the frame as the chessboard.
    Uses OCR on the candidate board region to determine orientation and passes
    the appropriate black_view parameter to get_fen_from_image.
    Returns the cropped FEN (board layout only) from get_fen_from_image.
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
        candidate_img = frame[y:y+h, x:x+w]
        # Determine orientation using OCR on candidate_img
        is_black = detect_black_perspective(candidate_img)
        orientation = "black" if is_black else "white"
        print(f"Detected orientation: {orientation}")
        # Call board_to_fen with black_view parameter accordingly
        pil_img = Image.fromarray(cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB))
        try:
            detected_fen = get_fen_from_image(pil_img, black_view=is_black).split(" ")[0]
            print(f"Detected FEN: {detected_fen}")
        except Exception as e:
            print(f"Error in FEN detection: {e}")
            detected_fen = None
        return detected_fen
    print("No chessboard detected in this frame.")
    return None

# === Stockfish Worker Thread ===
def stockfish_worker():
    """
    Worker thread that evaluates FENs using Stockfish.
    """
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        while True:
            fen = fen_queue.get()
            if fen is None:
                break
            try:
                board = chess.Board(fen)
                info = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
                eval_score = info["score"].white().score(mate_score=10000) / 100
                eval_results[fen] = eval_score
                print(f"Evaluated FEN: {fen} → Score: {eval_score}")
            except Exception as e:
                print(f"Stockfish error for FEN {fen}: {e}")
                eval_results[fen] = None  # Avoid crashes
            fen_queue.task_done()

def evaluate_fens_parallel(full_fen_list):
    """
    Evaluates all FENs in parallel using Stockfish, loading/saving evaluations to CSV.
    """
    global eval_results
    eval_results = load_evaluations()  # Load existing evaluations

    # Find FENs that still need evaluation
    fens_to_evaluate = [fen for fen in full_fen_list if fen not in eval_results]

    if fens_to_evaluate:
        workers = []
        for _ in range(NUM_WORKERS):
            t = threading.Thread(target=stockfish_worker, daemon=True)
            t.start()
            workers.append(t)

        for fen in fens_to_evaluate:
            fen_queue.put(fen)

        fen_queue.join()

        for _ in range(NUM_WORKERS):
            fen_queue.put(None)
        for t in workers:
            t.join()

        # Save the newly computed evaluations
        save_evaluations(eval_results)

    return eval_results

# === Process Video and Write SRT Incrementally ===
def process_video(video_path, expected_fens, evaluated_scores, output_file):
    """
    Processes the video and writes an SRT entry when a new FEN is detected.
    Each subtitle covers the period from when that FEN is first detected until just before the next FEN change.
    If the detected FEN matches one from the PGN (expected_fens), its evaluation is used.
    For the starting position, evaluation is forced to 0.0.
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
                        end_sec = current_sec
                        start_tc = format_timecode(current_subtitle_start_sec)
                        end_tc = format_timecode(end_sec)
                        bar = draw_eval_bar(current_eval)
                        numeric = f"[{fullwidth_eval(current_eval)}]"
                        srt_file.write(f"{subtitle_index}\n{start_tc} --> {end_tc}\n{bar} {numeric}\n\n")
                        srt_file.flush()
                        print(f"Subtitle {subtitle_index}: {start_tc} --> {end_tc} | {bar} {numeric}")
                        subtitle_index += 1
                    current_subtitle_fen = detected
                    current_subtitle_start_sec = current_sec
                    if detected == STANDARD_START:
                        new_eval = 0.0
                        print("Standard starting position detected; forcing eval to 0.0")
                    elif detected in expected_fens:
                        full_fen, move_num = expected_fens[detected]
                        new_eval = evaluated_scores.get(full_fen, 0.0)
                        print(f"New FEN (move {move_num}): {detected} with eval {new_eval:.2f}")
                    else:
                        print(f"New FEN: {detected} not in expected; using current_eval {current_eval:.2f}")
                        new_eval = current_eval
                    current_eval = new_eval
            else:
                print(f"Second {current_sec}: No FEN detected")
        frame_index += 1
    
    last_sec = frame_index // fps
    if current_subtitle_fen is not None and current_subtitle_start_sec is not None:
        start_tc = format_timecode(current_subtitle_start_sec)
        end_tc = format_timecode(last_sec)
        bar = draw_eval_bar(current_eval)
        numeric = f"[{fullwidth_eval(current_eval)}]"
        srt_file.write(f"{subtitle_index}\n{start_tc} --> {end_tc}\n{bar} {numeric}\n\n")
        srt_file.flush()
        print(f"Subtitle {subtitle_index}: {start_tc} --> {end_tc} | {bar} {numeric}")
    
    srt_file.close()
    cap.release()
    print("Video processing complete.")

def main():
    print("Starting Chess Video Analysis")
    with open(OUTPUT_SRT, "w") as f:
        f.write("")
    expected_fens = get_all_expected_fens()
    print(f"Extracted {len(expected_fens)} expected moves from PGN files.")
    full_fen_list = [full for full, _ in expected_fens.values()]
    evaluated_scores = evaluate_fens_parallel(full_fen_list)
    video_files = get_video_files()
    if not video_files:
        print("No .mp4 files found in the current directory.")
        return
    for video_file in video_files:
        print(f"Processing video: {video_file}")
        process_video(video_file, expected_fens, evaluated_scores, f"{os.path.splitext(video_file)[0]}.srt")
    print("Chess Video Analysis Completed.")

if __name__ == "__main__":
    main()
