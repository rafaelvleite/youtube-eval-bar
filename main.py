import chess
import chess.pgn
import chess.engine
import cv2
import threading
import queue
from PIL import Image
from board_to_fen.predict import get_fen_from_image
import os
import pytesseract
import csv
import concurrent.futures
import numpy as np

# Import the new API function from your external module.
from highlighted_squares_detection import detect_turn

# === CONFIGURATION ===
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
STOCKFISH_DEPTH = 18            # Evaluation depth
NUM_WORKERS = os.cpu_count() or 4
FRAME_INTERVAL = 60             # Process one frame per second (assuming 60 FPS)
EVAL_CSV_FILE = "evaluations.csv"

# === GLOBALS ===
fen_queue = queue.Queue()
eval_results = {}

# Standard starting position (piece placement only)
STANDARD_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

############################################################
#                  HELPER FUNCTIONS (GENERAL)
############################################################

def get_video_files():
    """Detect all .mp4 files in the current directory."""
    return [f for f in os.listdir(".") if f.lower().endswith(".mp4")]

def load_evaluations():
    """Load previously computed evaluations from a CSV file."""
    evaluations = {}
    if os.path.exists(EVAL_CSV_FILE):
        with open(EVAL_CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    evaluations[row[0]] = float(row[1])
    print(f"Loaded {len(evaluations)} evaluations from CSV.")
    return evaluations

def save_evaluations(evaluations):
    """Save evaluations to a CSV file."""
    with open(EVAL_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for fen, score in evaluations.items():
            writer.writerow([fen, score])
    print(f"Saved {len(evaluations)} evaluations to CSV.")

def format_timecode(seconds):
    """Convert seconds to an SRT timecode."""
    mins, secs = divmod(int(seconds), 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"00:{mins:02}:{secs:02},{ms:03}"

def draw_eval_bar(eval_score):
    """Draw a bar using 24 characters to represent evaluation score."""
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

def fullwidth_eval(eval_score):
    """Convert numeric evaluation to fullwidth symbols, or return 'Mate vindo' if eval exceeds threshold."""
    if eval_score > 20 or eval_score < -20:
        return "Mate vindo"

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

############################################################
#              PGN PROCESSING & EXPECTED FENS
############################################################

def get_all_expected_fens():
    """
    Reads .pgn files to extract FENs from mainline and variations.
    Positions are stored as keys: (arrangement, side).
    """
    expected = {}
    for filename in os.listdir("."):
        if filename.lower().endswith(".pgn"):
            with open(filename, "r", encoding="latin-1") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    stack = [(game, chess.Board(), 0)]  # (node, board, move_num)
                    while stack:
                        node, board, move_num = stack.pop()
                        if node.move is not None:
                            try:
                                board.push(node.move)
                                move_num += 1
                                full_fen = board.fen()  # e.g., "rnbqkbnr/ppp... b KQkq - 0 1"
                                parts = full_fen.split()
                                if len(parts) >= 2:
                                    arrangement = parts[0]
                                    side = parts[1]
                                    key = (arrangement, side)
                                    if key not in expected:
                                        expected[key] = (full_fen, move_num)
                            except Exception as e:
                                print(f"Warning: Skipping illegal move {node.move} in {board.fen()}")
                                continue
                        for var in node.variations:
                            stack.append((var, board.copy(), move_num))
    print(f"Extracted {len(expected)} unique positions from PGN files.")
    return expected

############################################################
#            BLACK ORIENTATION DETECTION
############################################################

def detect_black_perspective(roi, debug=False):
    """
    Looks at the bottom-left corner of `roi` to see if it detects 'a1' (white's perspective)
    or 'a8' (black's perspective). Returns True if black perspective.
    """
    h, w = roi.shape[:2]
    crop_size = 30
    corner_bl = roi[h - crop_size:h, 0:crop_size]

    def ocr_corner(image):
        scale_factor = 2.0
        new_w = int(image.shape[1] * scale_factor)
        new_h = int(image.shape[0] * scale_factor)
        img_up = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        filtered = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        config_str = r"--psm 10 -c tessedit_char_whitelist=abcdefgh12345678"
        text = pytesseract.image_to_string(filtered, config=config_str).strip().lower()
        return text

    text_bl = ocr_corner(corner_bl).replace("\n", "")
    if debug:
        print(f"[DEBUG] Bottom-left OCR => '{text_bl}'")
    if 'a8' in text_bl:
        return True
    elif 'a1' in text_bl:
        return False
    else:
        if '8' in text_bl and 'a' in text_bl:
            return True
        elif 'a' in text_bl or '1' in text_bl:
            return False
        elif '8' in text_bl:
            return True
    return False

############################################################
#              CHESSBOARD DETECTION (UPDATED)
############################################################

def detect_chessboard(frame):
    """
    Locate the chessboard in the frame, correct perspective,
    get the partial FEN, and detect turn using the external API.
    Returns a dict with keys:
       - "partial_fen": the piece placement string,
       - "turn_info": turn information.
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
        is_black = detect_black_perspective(candidate_img)
        orientation = "black" if is_black else "white"
        print(f"Detected orientation: {orientation}")

        try:
            pil_img = Image.fromarray(cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB))
            detected_fen = get_fen_from_image(pil_img, black_view=is_black)
            partial_fen = detected_fen.split()[0]
            print(f"Detected partial FEN: {partial_fen}")
        except Exception as e:
            print(f"Error in FEN detection: {e}")
            return None

        # Use the external API for turn detection.
        if detected_fen.startswith(STANDARD_START):
            turn_info = "White's turn"  # Força o turno para a posição inicial
        else:
            turn_info = detect_turn(candidate_img)        
        print(f"Turn detection result: {turn_info}")
        return {"partial_fen": partial_fen, "turn_info": turn_info}

    print("No chessboard detected in this frame.")
    return None

############################################################
#         STOCKFISH EVALUATION WORKER
############################################################

def stockfish_worker():
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
                eval_results[fen] = None
            fen_queue.task_done()

def evaluate_fens_parallel(full_fen_list):
    """Evaluate FENs in parallel using worker threads and update CSV evaluations."""
    global eval_results
    eval_results = load_evaluations()
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
        save_evaluations(eval_results)
    return eval_results

############################################################
#         VIDEO PROCESSING + SUBTITLES
############################################################

def process_video_multiprocessing(video_path, expected_fens, evaluated_scores, output_file):
    """
    Processa o vídeo frame a frame. A cada FRAME_INTERVAL, detecta o tabuleiro,
    reconstrói a FEN e obtém a avaliação e informações de turno.
    Gera segmentos SRT com barras de avaliação e detalhes de turno.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing Video: {video_path} ({total_frames} frames, {fps} FPS)")

    tasks = []  # Cada tarefa é uma tupla: (frame_index, segundo, future)
    frame_index = 0
    current_subtitle_fen = None
    current_subtitle_start_sec = None
    current_eval = 0.0
    current_turn = "Unknown"
    subtitle_index = 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % FRAME_INTERVAL == 0:
                current_sec = frame_index // fps
                future = executor.submit(detect_chessboard, frame)
                tasks.append((frame_index, current_sec, future))
            frame_index += 1
    cap.release()

    with open(output_file, "w", encoding="utf-8") as srt_file:
        for frame_idx, sec, future in tasks:
            result = future.result()
            if result:
                partial_fen = result.get("partial_fen")
                turn_info = result.get("turn_info")
                # Se a posição for a inicial, força o turno para as brancas.
                if partial_fen.startswith(STANDARD_START):
                    turn_info = "White's turn"
                desired_side = "w" if turn_info == "White's turn" else "b" if turn_info == "Black's turn" else None
                if partial_fen:
                    key = (partial_fen, desired_side)
                    new_fen = None
                    if key in expected_fens:
                        new_fen, move_num = expected_fens[key]
                    else:
                        print(f"Key {key} not found in expected_fens.")
                    if new_fen is not None:
                        print(f"Second {sec}: Reconstructed full FEN: {new_fen}")
                        if new_fen != current_subtitle_fen:
                            # Se já existe um segmento anterior, finaliza o segmento atual
                            if current_subtitle_fen and current_subtitle_start_sec is not None:
                                end_sec = sec
                                start_tc = format_timecode(current_subtitle_start_sec)
                                end_tc = format_timecode(end_sec)
                                bar = draw_eval_bar(current_eval)
                                numeric = f"[{fullwidth_eval(current_eval)}]".replace("[Mate vindo]", "Mate vindo")
                                srt_file.write(
                                    f"{subtitle_index}\n{start_tc} --> {end_tc}\n{bar} {numeric}\n\n"
                                )
                                srt_file.flush()
                                print(f"Subtitle {subtitle_index}: {start_tc} --> {end_tc} | {bar} {numeric}")
                                subtitle_index += 1
                            # Atualiza a posição atual e a hora de início do novo segmento
                            current_subtitle_fen = new_fen
                            current_subtitle_start_sec = sec
                            parts = new_fen.split()
                            if len(parts) >= 2:
                                arrangement = parts[0]
                                side = parts[1]
                                key_eval = (arrangement, side)
                                if arrangement == STANDARD_START and side == "w":
                                    new_eval = 0.0
                                    print("Standard start => forced eval 0.0")
                                elif key_eval in expected_fens:
                                    full_fen, move_num = expected_fens[key_eval]
                                    new_eval = evaluated_scores.get(full_fen, 0.0)
                                    print(f"Found expected FEN (move {move_num}): {full_fen} => eval {new_eval:.2f}")
                                else:
                                    print(f"Arrangement+side {key_eval} not in expected => reusing {current_eval:.2f}")
                                    new_eval = current_eval
                                current_eval = new_eval
            else:
                print(f"Second {sec}: No board detected.")

        last_sec = frame_index // fps
        # Gera o último segmento, sem condição para subtitle_index
        if current_subtitle_fen and current_subtitle_start_sec is not None:
            start_tc = format_timecode(current_subtitle_start_sec)
            end_tc = format_timecode(last_sec)
            bar = draw_eval_bar(current_eval)
            numeric = f"[{fullwidth_eval(current_eval)}]"
            srt_file.write(
                f"{subtitle_index}\n{start_tc} --> {end_tc}\n{bar} {numeric}\n\n"
            )
            srt_file.flush()
            print(f"Subtitle {subtitle_index}: {start_tc} --> {end_tc} | {bar} {numeric}")

    print("Video processing complete.")

############################################################
#                  MAIN FUNCTION
############################################################

def main():
    print("Starting Chess Video Analysis")
    # 1) Collect expected positions from PGN.
    expected_fens = get_all_expected_fens()
    # Adiciona a posição inicial, se não estiver presente
    if (STANDARD_START, "w") not in expected_fens:
        # Gera uma FEN completa para a posição inicial, considerando a vez das brancas e a disponibilidade de roques.
        expected_fens[(STANDARD_START, "w")] = (f"{STANDARD_START} w KQkq - 0 1", 0)
    print(f"Extracted {len(expected_fens)} unique positions from PGN files.")

    # 2) Evaluate those positions.
    full_fen_list = [val[0] for val in expected_fens.values()]
    evaluated_scores = evaluate_fens_parallel(full_fen_list)

    # 3) Process each .mp4 file in the directory.
    video_files = get_video_files()
    if not video_files:
        print("No .mp4 files found in the current directory.")
        return

    for video_file in video_files:
        srt_file = f"{os.path.splitext(video_file)[0]}.srt"
        if os.path.exists(srt_file):
            print(f"Skipping {video_file} (SRT file already exists: {srt_file})")
            continue
        print(f"Processing video: {video_file}")
        process_video_multiprocessing(video_file, expected_fens, evaluated_scores, srt_file)

    print("Chess Video Analysis Completed.")

if __name__ == "__main__":
    main()
