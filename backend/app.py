from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import datetime
import os
import numpy as np
from collections import Counter, deque
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 

# --- 1. Load Models ---
print("Loading Models...")
plate_model = YOLO("models/plate_detector.pt")
char_model = YOLO("models/char_detector.pt")
print("Models Loaded!")

# --- Settings ---
CONF_THRESHOLD = 0.5
latest_log = []
current_source = None 
is_paused = False

HISTORY_LEN = 10  
top_line_buffer = deque(maxlen=HISTORY_LEN)
bottom_line_buffer = deque(maxlen=HISTORY_LEN)
last_logged_text = ""
last_log_time = datetime.datetime.min 

# --- 2. Mapping (อัปเดตตาม list ที่ให้มา) ---
CHAR_MAP = {
    # ตัวเลข 0-9
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    
    # พยัญชนะ (A01 - A44)
    'A01': 'ก', 'A02': 'ข', 'A03': 'ฃ', 'A04': 'ค', 'A05': 'ฅ', 
    'A06': 'ฆ', 'A07': 'ง', 'A08': 'จ', 'A09': 'ฉ', 'A10': 'ช',
    'A11': 'ซ', 'A12': 'ฌ', 'A13': 'ญ', 'A14': 'ฎ', 'A15': 'ฏ',
    'A16': 'ฐ', 'A17': 'ฑ', 'A18': 'ฒ', 'A19': 'ณ', 'A20': 'ด',
    'A21': 'ต', 'A22': 'ถ', 'A23': 'ท', 'A24': 'ธ', 'A25': 'น',
    'A26': 'บ', 'A27': 'ป', 'A28': 'ผ', 'A29': 'ฝ', 'A30': 'พ',
    'A31': 'ฟ', 'A32': 'ภ', 'A33': 'ม', 'A34': 'ย', 'A35': 'ร',
    'A36': 'ล', 'A37': 'ว', 'A38': 'ศ', 'A39': 'ษ', 'A40': 'ส',
    'A41': 'ห', 'A42': 'ฬ', 'A43': 'อ', 'A44': 'ฮ',

    # จังหวัด (อ้างอิงจาก List ที่ให้มา)
    'ACR': 'อำนาจเจริญ',    'ATG': 'อ่างทอง',       'AYA': 'พระนครศรีอยุธยา',
    'BKK': 'กรุงเทพมหานคร', 'BKN': 'บึงกาฬ',        'BRM': 'บุรีรัมย์',
    'BTG': 'เบตง',          'CBI': 'ชลบุรี',        'CCO': 'ฉะเชิงเทรา',
    'CMI': 'เชียงใหม่',     'CNT': 'ชัยนาท',        'CPM': 'ชัยภูมิ',
    'CPN': 'ชุมพร',         'CRI': 'เชียงราย',      'CTI': 'จันทบุรี',
    'KBI': 'กระบี่',        'KKN': 'ขอนแก่น',       'KPT': 'กำแพงเพชร',
    'KRI': 'กาญจนบุรี',     'KSN': 'กาฬสินธุ์',     'LEI': 'เลย',
    'LPG': 'ลำปาง',         'LPN': 'ลำพูน',         'LRI': 'ลพบุรี',
    'MDH': 'มุกดาหาร',      'MKM': 'มหาสารคาม',     'MSN': 'แม่ฮ่องสอน',
    'NAN': 'น่าน',          'NBI': 'นนทบุรี',       'NBP': 'หนองบัวลำภู',
    'NKI': 'หนองคาย',       'NMA': 'นครราชสีมา',    'NPM': 'นครพนม',
    'NPT': 'นครปฐม',        'NSN': 'นครสวรรค์',     'NST': 'นครศรีธรรมราช',
    'NYK': 'นครนายก',       'PBI': 'ปราจีนบุรี',    'PCT': 'พิจิตร',
    'PKN': 'ประจวบคีรีขันธ์', 'PKT': 'ภูเก็ต',      'PLG': 'พัทลุง',
    'PLK': 'พิษณุโลก',      'PNA': 'พังงา',         'PNB': 'เพชรบูรณ์',
    'PRE': 'แพร่',          'PRI': 'เพชรบุรี',      'PTE': 'ปทุมธานี',
    'PTN': 'ปัตตานี',       'PYO': 'พะเยา',         'RBR': 'ราชบุรี',
    'RET': 'ร้อยเอ็ด',      'RNG': 'ระนอง',         'RYG': 'ระยอง',
    'SBR': 'สระบุรี',       'SKA': 'สงขลา',         'SKM': 'สมุทรสงคราม',
    'SKN': 'สมุทรสาคร',     'SKW': 'สระแก้ว',       'SNI': 'สิงห์บุรี',
    'SNK': 'สกลนคร',        'SPB': 'สุพรรณบุรี',    'SPK': 'สมุทรปราการ',
    'SRI': 'สุราษฎร์ธานี',  'SRN': 'สุรินทร์',      'SSK': 'ศรีสะเกษ',
    'STI': 'สุโขทัย',       'TAK': 'ตาก',           'TRG': 'ตรัง',
    'TRT': 'ตราด',          'UBN': 'อุบลราชธานี',   'UDN': 'อุดรธานี',
    'UTI': 'อุทัยธานี',     'UTT': 'อุตรดิตถ์',     'YLA': 'ยะลา',
    'YST': 'ยโสธร'
}

def decode_label(class_name):
    return CHAR_MAP.get(class_name, class_name)

# --- 3. Logic แยกบรรทัด ---
def split_lines(detections):
    if not detections: return None, None
    detections.sort(key=lambda x: x['y_center'])
    min_y = min(d['y_center'] for d in detections)
    max_y = max(d['y_center'] for d in detections)
    
    line1, line2 = [], []
    if (max_y - min_y) < 30:
        line1 = detections
    else:
        y_midpoint = min_y + (max_y - min_y) * 0.5
        for d in detections:
            if d['y_center'] < y_midpoint:
                line1.append(d)
            else:
                line2.append(d)
    line1.sort(key=lambda x: x['x1'])
    line2.sort(key=lambda x: x['x1'])
    text_top = "".join([decode_label(d['name']) for d in line1])
    text_bottom = "".join([decode_label(d['name']) for d in line2])
    return text_top, text_bottom

# 🚨 [อัปเกรด] ต้องโหวตชนะเกิน 15 เสียง (50%) ถึงจะผ่าน
def get_best_text(buffer):
    if not buffer: return ""
    counts = Counter(buffer)
    most_common = counts.most_common(1)[0] 
    text, count = most_common
    
    # ต้องนิ่งจริง (เจอซ้ำๆ เกิน 7 ครั้ง ใน 30 เฟรม)
    if count >= 7: 
        return text
    return ""

def draw_thai_text(img, text, position, font_size=30, color=(0, 255, 0)):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font_path = "C:/Windows/Fonts/tahoma.ttf" 
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle(bbox, fill="black") 
        draw.text(position, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        return img

# --- Generator ---
def generate_frames():
    global latest_log, current_source, is_paused, last_logged_text, last_log_time
    
    # ... (ส่วนรอวิดีโอ เหมือนเดิม) ...
    while current_source is None:
        # ... (โค้ดแสดงหน้าจอดำ) ...
        yield (...)
        cv2.waitKey(100)

    cap = cv2.VideoCapture(current_source)
    frame_count = 0
    last_detections = []
    last_frame_buffer = None
    
    # เคลียร์ค่าเริ่มต้น
    top_line_buffer.clear()
    bottom_line_buffer.clear()
    last_logged_text = ""
    
    # 🚨 [ใหม่] ตัวนับความว่างเปล่า
    no_plate_count = 0 

    while cap.isOpened():
        # ... (ส่วน Pause เหมือนเดิม) ...
        if is_paused:
            # ...
            continue

        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        
        # Scan ทุก 3 เฟรม
        if frame_count % 3 == 0:
            current_frame_detections = [] 
            try:
                plate_results = plate_model(frame, conf=CONF_THRESHOLD)
                
                # 🚨 [ใหม่] เช็คว่าเจอป้ายไหม?
                # ถ้าไม่เจอเลย (len == 0) ให้เริ่มนับถอยหลัง
                found_any_plate = False
                for res in plate_results:
                    if len(res.boxes) > 0:
                        found_any_plate = True
                        break
                
                if not found_any_plate:
                    no_plate_count += 1
                    # ถ้าไม่เจอป้ายติดต่อกัน 10 รอบสแกน (ประมาณ 30 เฟรม = 1 วินาที)
                    if no_plate_count > 10:
                        top_line_buffer.clear()     # ล้างความจำบรรทัดบน
                        bottom_line_buffer.clear()  # ล้างความจำบรรทัดล่าง
                        last_logged_text = ""       # ล้างกันซ้ำ
                        # print("Clear Buffer!")    # (เอาไว้เช็คได้)
                else:
                    # ถ้าเจอ ให้รีเซ็ตตัวนับ
                    no_plate_count = 0

                # --- (เข้าสู่ลูปเดิม) ---
                for res in plate_results:
                    for box in res.boxes:
                        # ... (โค้ดเดิมทั้งหมด: Crop, Padding, Char Model, Split lines) ...
                        px1, py1, px2, py2 = [int(i) for i in box.xyxy[0]]
                        try:
                            # ... (Padding) ...
                            h, w, _ = frame.shape
                            padding = 15 
                            px1_pad = max(0, px1 - padding)
                            py1_pad = max(0, py1 - padding)
                            px2_pad = min(w, px2 + padding)
                            py2_pad = min(h, py2 + padding)

                            plate_img = frame[py1_pad:py2_pad, px1_pad:px2_pad]
                            char_results = char_model(plate_img, conf=0.4)
                            
                            char_list = []
                            if char_results[0]:
                                for cbox in char_results[0].boxes:
                                    # ... (เก็บ char_list) ...
                                    cx1, cy1, cx2, cy2 = [int(i) for i in cbox.xyxy[0]]
                                    cls_id = int(cbox.cls[0])
                                    class_name = char_model.names[cls_id]
                                    char_list.append({
                                        'name': class_name, 
                                        'x1': cx1, 'y_center': (cy1 + cy2) / 2
                                    })
                            
                            t_top, t_bottom = split_lines(char_list)
                            
                            # 🚨 เติม Buffer (เฉพาะตอนที่เจอตัวหนังสือ)
                            if t_top: top_line_buffer.append(t_top)
                            if t_bottom: bottom_line_buffer.append(t_bottom)
                            
                            # ... (Logic เดิม: Vote, Log) ...
                            best_top = get_best_text(top_line_buffer)
                            best_bottom = get_best_text(bottom_line_buffer)
                            full_text = f"{best_top} {best_bottom}".strip()
                            
                            if full_text and len(best_top) >= 2:
                                current_frame_detections.append((px1, py1, px2, py2, full_text))
                                
                                # Log Logic
                                now = datetime.datetime.now()
                                is_substring = (full_text in last_logged_text) or (last_logged_text in full_text)
                                time_diff = (now - last_log_time).total_seconds() if last_log_time != datetime.datetime.min else 999

                                if (full_text != last_logged_text and not is_substring) or (time_diff > 10):
                                    if len(full_text.replace(" ", "")) > 3:
                                        timestamp = now.strftime("%H:%M:%S")
                                        latest_log.insert(0, {"time": timestamp, "text": full_text})
                                        latest_log = latest_log[:15]
                                        last_logged_text = full_text
                                        last_log_time = now
                        except: pass
                
                # อัปเดตการวาดกรอบ
                if found_any_plate:
                    last_detections = current_frame_detections
                else:
                    if no_plate_count > 5: # ถ้าไม่เจอแป๊บเดียว ให้ลบกรอบออกเลย
                        last_detections = []

            except: pass

        # ... (ส่วน Draw Boxes และ Encode Image เหมือนเดิม) ...
        for (x1, y1, x2, y2, text) in last_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = draw_thai_text(frame, text, (x1, y1 - 40))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        last_frame_buffer = frame
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

# --- Routes (เหมือนเดิม) ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_logs')
def get_logs():
    return jsonify(latest_log)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_source, is_paused
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file"}), 400
    
    filename = file.filename
    ext = os.path.splitext(filename)[1]
    
    filepath = os.path.join("temp_video" + ext)
    file.save(filepath)
    
    current_source = filepath
    is_paused = False
    top_line_buffer.clear()
    bottom_line_buffer.clear()
    return jsonify({"message": "Video uploaded"})

@app.route('/toggle_playback', methods=['POST'])
def toggle_playback():
    global is_paused
    is_paused = not is_paused
    return jsonify({"is_paused": is_paused})

if __name__ == "__main__":
    app.run(debug=True, port=5001)