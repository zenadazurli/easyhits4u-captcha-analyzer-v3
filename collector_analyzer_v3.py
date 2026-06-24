#!/usr/bin/env python3
# collector_analyzer_v3.py
# Raccoglie captcha con figure ritagliate e etichette
# Gestisce figure e matematici con risolutore integrato

import os
import sys
import time
import threading
import random
import requests
import json
import numpy as np
import cv2
import re
from datetime import datetime
from supabase import create_client
from datasets import load_dataset
import urllib3
import ddddocr
import easyocr

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== CONFIGURAZIONE ====================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
COOKIE_SUPABASE_URL = os.environ.get("COOKIE_SUPABASE_URL")
COOKIE_SUPABASE_KEY = os.environ.get("COOKIE_SUPABASE_KEY")
BUCKET_NAME = "easyhits4u-captchas-analyzer"
DATASET_REPO = "zenadazurli/easyhits4u-dataset"
DIM = 64
REQUEST_TIMEOUT = 15

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 3))
STAGGERED_START_DELAY = int(os.environ.get("STAGGERED_START_DELAY", 5))
REFRESH_INTERVAL = 1200  # Refresh sessione ogni 20 minuti

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_URL e SUPABASE_KEY devono essere impostate")
if not COOKIE_SUPABASE_URL or not COOKIE_SUPABASE_KEY:
    raise ValueError("❌ COOKIE_SUPABASE_URL e COOKIE_SUPABASE_KEY devono essere impostate")

# ==================== PROXY ====================
PROXY_LIST = [
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:13822",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:14693",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:13711",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:14329",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:14012",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:14465",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:13768",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:13506",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:14995",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:13353",
    "sazz16014w96:t3vz152mql23@resi.fusionproxy.net:13231"
]

# ==================== ACCOUNT ====================
NUOVI_ACCOUNT = [
    'unobbufjagl', 'ucutuva', 'ucufrkrreea', 'uvofichiad',
    'usazobe', 'uenbebe', 'upabesaki', 'uwachikifebb',
    'uremmnasama', 'ukovolece', 'udidadituchi', 'udidabbkane',
    'Giovannixxo', 'upaooli', 'unojuquvu', 'usffotalobb',
    'uleookipg', 'Pippopizza', 'utrfiooadrm', 'ukrtaorlibe',
    'ukidipa', 'ufivora', 'ukaootamotr', 'uvubbufoo',
    'ujanefemi', 'uxizoread', 'urzloufbb', 'uchijufror',
    'uvuncrm', 'uaatavokirz', 'urmmmlasfda', 'unegavoaaga',
    'uchioosachi', 'uookuvafebo', 'utapgwabe', 'ukrbolutu',
    'ufofrpazu', 'uqudaadtu', 'uooufcetuza', 'uzunotalu'
]

# ==================== VARIABILI GLOBALI ====================
X_fast = None
y_fast = None
classes_fast = None
proxy_index = 0
proxy_lock = threading.Lock()
dddd_ocr = None
easy_reader = None

# ==================== FUNZIONI DATASET ====================
def load_dataset_from_hf():
    """Carica il dataset da Hugging Face"""
    global X_fast, y_fast, classes_fast
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📥 Caricamento dataset da Hugging Face: {DATASET_REPO}", flush=True)
    
    try:
        dataset = load_dataset(DATASET_REPO, trust_remote_code=True)
        data = dataset.get("train") if "train" in dataset else dataset
        
        X = []
        y = []
        class_to_idx = {}
        
        for item in data:
            features = item.get("X")
            label_idx = item.get("y")
            if features is None or label_idx is None:
                continue
            
            if hasattr(data.features['y'], 'names'):
                class_name = data.features['y'].names[label_idx]
            else:
                class_name = str(label_idx)
            
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
            
            X.append(np.array(features, dtype=np.float32))
            y.append(class_to_idx[class_name])
        
        if X:
            X_fast = np.vstack(X).astype(np.float32)
            y_fast = np.array(y, dtype=np.int32)
            classes_fast = {v: k for k, v in class_to_idx.items()}
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Dataset caricato: {X_fast.shape[0]} vettori, {len(classes_fast)} classi", flush=True)
            return True
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Nessun dato valido nel dataset", flush=True)
            return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Errore caricamento dataset: {e}", flush=True)
        return False

# ==================== FUNZIONI RISOLUTORE MATEMATICO ====================
def init_math_ocr():
    """Inizializza OCR per captcha matematici"""
    global dddd_ocr, easy_reader
    try:
        dddd_ocr = ddddocr.DdddOcr()
        dddd_ocr.set_ranges("0123456789+-")
        easy_reader = easyocr.Reader(['en'], gpu=False)
        log("✅ OCR matematici inizializzati")
        return True
    except Exception as e:
        log(f"❌ Errore inizializzazione OCR: {e}")
        return False

def risolvi_captcha_matematico(img, surfses):
    """
    Risolve captcha matematico con denoise elastico
    Restituisce: (segno, risultato, a, b) o (None, None, None, None)
    """
    if img is None or dddd_ocr is None:
        return None, None, None, None
    
    # Assicurati che sia in scala di grigi
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    migliore = None
    miglior_punteggio = -1
    
    # Denoise elastico (3-30, step 2)
    for forza in range(3, 31, 2):
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, forza, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Estrai caratteri con coordinate
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        caratteri = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 20 or area > 500:
                continue
            
            roi = binary[y:y+h, x:x+w]
            roi = cv2.copyMakeBorder(roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
            _, buffer = cv2.imencode('.png', roi)
            risultato = dddd_ocr.classification(buffer.tobytes())
            if risultato and risultato in "0123456789+-":
                caratteri.append({'c': risultato, 'x': x, 'w': w})
        
        if len(caratteri) < 2:
            continue
        
        caratteri.sort(key=lambda c: c['x'])
        
        # Cerca segno
        segno = None
        for c in caratteri:
            if c['c'] in ['+', '-']:
                segno = c['c']
                break
        
        # Separa numeri con segno
        if segno:
            numeri = []
            corrente = []
            for c in caratteri:
                if c['c'] == segno:
                    if corrente:
                        numeri.append(int(''.join(corrente)))
                        corrente = []
                elif c['c'].isdigit():
                    corrente.append(c['c'])
            if corrente:
                numeri.append(int(''.join(corrente)))
            
            if len(numeri) >= 2:
                punteggio = numeri[0] + numeri[1]
                if punteggio > miglior_punteggio:
                    miglior_punteggio = punteggio
                    migliore = (numeri[0], segno, numeri[1])
                continue
        
        # Separa per distanza (se il segno manca)
        numeri = []
        corrente = []
        ultima_x = caratteri[0]['x']
        for c in caratteri:
            if not c['c'].isdigit():
                continue
            if c['x'] - ultima_x > 30:
                if corrente:
                    numeri.append(int(''.join(corrente)))
                    corrente = []
            corrente.append(c['c'])
            ultima_x = c['x'] + c['w']
        if corrente:
            numeri.append(int(''.join(corrente)))
        
        if len(numeri) >= 2:
            punteggio = numeri[0] + numeri[1]
            if punteggio > miglior_punteggio:
                miglior_punteggio = punteggio
                migliore = (numeri[0], '+', numeri[1])
    
    # Se DdddOcr fallisce, prova EasyOCR
    if migliore is None and easy_reader is not None:
        try:
            risultato = easy_reader.readtext(gray, detail=0)
            testo = ' '.join(risultato)
            if testo:
                # Converti parole in numeri
                word_to_num = {
                    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
                    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
                    'eighteen': '18', 'nineteen': '19', 'twenty': '20',
                    'plus': '+', 'minus': '-', 'and': '+'
                }
                for word, num in word_to_num.items():
                    testo = testo.lower().replace(word, str(num))
                
                match = re.search(r'(\d+)\s*([+\-])\s*(\d+)', testo)
                if match:
                    migliore = (int(match.group(1)), match.group(2), int(match.group(3)))
        except Exception as e:
            log(f"⚠️ Errore EasyOCR: {e}")
    
    if migliore is None:
        return None, None, None, None
    
    a, segno, b = migliore
    
    # Calcola entrambi i risultati
    r_plus = a + b
    r_minus = a - b
    
    # Usa i dati del server per scegliere
    opzioni = []
    for key in ['aword1_number', 'aword2_number', 'aword3_number']:
        val = surfses.get(key)
        if val is not None:
            opzioni.append(int(val))
    
    if opzioni:
        if r_plus in opzioni:
            return '+', r_plus, a, b
        elif r_minus in opzioni:
            return '-', r_minus, a, b
    
    # Se il server non aiuta, usa il segno trovato
    if segno in ['+', '-']:
        return segno, a + b if segno == '+' else a - b, a, b
    
    return '+', r_plus, a, b

def numero_a_parola(num):
    """Converte numero in parola inglese"""
    mappa = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
        6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
        11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
        15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
        19: 'nineteen', 20: 'twenty'
    }
    return mappa.get(num, str(num))

# ==================== FUNZIONI PROXY ====================
def get_next_proxy():
    """Restituisce il prossimo proxy in rotazione (round-robin)"""
    global proxy_index
    with proxy_lock:
        proxy = PROXY_LIST[proxy_index % len(PROXY_LIST)]
        proxy_index += 1
        return proxy

# ==================== FUNZIONI FIGURE ====================
def centra_figura(image):
    """Centra e ritaglia la figura"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (DIM, DIM))
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return cv2.resize(crop, (DIM, DIM))

def estrai_descrittori(img):
    """Estrae descrittori per la figura"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circularity = 0.0
    aspect_ratio = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if peri != 0:
            circularity = 4.0 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h != 0 else 0.0
    
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten().tolist()
    
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    raggi = [int(min(h,w)*r) for r in (0.2, 0.4, 0.6, 0.8)]
    radiale = []
    for r in raggi:
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (cx,cy), r, 255, -1)
        mean = cv2.mean(img, mask=mask)[:3]
        radiale.extend([m/255.0 for m in mean])
    
    spaziale = []
    quadranti = [(0,0,cx,cy), (cx,0,w,cy), (0,cy,cx,h), (cx,cy,w,h)]
    for (x1,y1,x2,y2) in quadranti:
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            mean = cv2.mean(roi)[:3]
            spaziale.extend([m/255.0 for m in mean])
    
    return radiale + spaziale + [circularity, aspect_ratio] + hu

def predict_figure(img_crop):
    """Riconosce una figura usando il dataset"""
    global X_fast, y_fast, classes_fast
    
    if X_fast is None or img_crop is None or img_crop.size == 0:
        return None
    
    img_centrata = centra_figura(img_crop)
    features = np.array(estrai_descrittori(img_centrata), dtype=float)
    distances = np.linalg.norm(X_fast - features, axis=1)
    best_idx = np.argmin(distances)
    return classes_fast.get(int(y_fast[best_idx]), None)

def crop_safe(img, coords):
    """Ritaglia in sicurezza dalle coordinate"""
    try:
        x1, y1, x2, y2 = map(int, coords.split(","))
    except:
        return None
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop

# ==================== SALVATAGGIO CAPTCHA ====================
def salva_captcha_analyzer(supabase_client, account_name, qpic, img, picmap, labels, motivo, urlid, stats):
    """Salva il captcha con figure ritagliate e etichette"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        
        if picmap is not None:
            prefix = "figure"
            table = "figure_captchas_analyzer"
            stats['figure'] += 1
            
            folder_name = f"{prefix}/{timestamp}_{account_name}"
            
            # 1. Salva immagine intera
            file_path = f"{folder_name}/full.png"
            _, buffer = cv2.imencode('.png', img)
            img_bytes = buffer.tobytes()
            supabase_client.storage.from_(BUCKET_NAME).upload(file_path, img_bytes)
            
            # 2. Salva le 5 figure ritagliate
            crop_paths = []
            crop_labels = []
            for i, p in enumerate(picmap):
                if i >= 5:
                    break
                crop = crop_safe(img, p.get("coords", ""))
                if crop is not None and crop.size > 0:
                    label = labels[i] if i < len(labels) else "unknown"
                    crop_filename = f"{folder_name}/crop_{i+1}_{label}.png"
                    _, crop_buffer = cv2.imencode('.png', crop)
                    supabase_client.storage.from_(BUCKET_NAME).upload(crop_filename, crop_buffer.tobytes())
                    crop_paths.append(crop_filename)
                    crop_labels.append(label)
            
            # 3. Salva metadati
            data = {
                'account_name': account_name,
                'image_path': file_path,
                'crop_paths': json.dumps(crop_paths),
                'labels': json.dumps(crop_labels),
                'picmap_data': json.dumps(picmap),
                'timestamp': datetime.now().isoformat(),
                'status': 'unsolved',
                'motivo': motivo,
                'urlid': urlid,
                'qpic': qpic
            }
            
        else:
            # Captcha matematico
            prefix = "math"
            table = "math_captchas_analyzer"
            stats['math'] += 1
            
            # Salva l'immagine se disponibile
            if img is not None:
                file_path = f"{prefix}/{timestamp}_{account_name}.png"
                _, buffer = cv2.imencode('.png', img)
                img_bytes = buffer.tobytes()
                supabase_client.storage.from_(BUCKET_NAME).upload(file_path, img_bytes)
            else:
                file_path = None
            
            data = {
                'account_name': account_name,
                'image_path': file_path,
                'timestamp': datetime.now().isoformat(),
                'status': 'unsolved',
                'motivo': motivo,
                'urlid': urlid,
                'qpic': qpic
            }
        
        supabase_client.table(table).insert(data).execute()
        log(f"[{account_name}] 💾 Captcha salvato ({motivo})")
        return True
    except Exception as e:
        log(f"[{account_name}] ❌ Errore salvataggio: {e}")
        return False

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

# ==================== SURF ACCOUNT ====================
def surf_account(account_name, cookie_string, stats, supabase_client):
    """Esegue surf per un account con refresh periodico della sessione e proxy"""
    
    def init_session(proxy=None):
        """Crea una sessione con header realistici e proxy opzionale"""
        session = requests.Session()
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        session.headers.update(headers)
        session.headers.update({"Cookie": cookie_string})
        
        if proxy:
            proxy_url = f"http://{proxy}"
            session.proxies = {
                "http": proxy_url,
                "https": proxy_url
            }
            log(f"[{account_name}] 🌐 Proxy: {proxy.split('@')[1] if '@' in proxy else proxy}")
        
        try:
            log(f"[{account_name}] 🔄 Attivazione sessione surf...")
            session.get("https://www.easyhits4u.com/surf/", verify=False, timeout=10)
            time.sleep(2)
        except Exception as e:
            log(f"[{account_name}] ⚠️ Errore attivazione surf: {e}")
        
        return session
    
    proxy = get_next_proxy()
    log(f"[{account_name}] 🌐 Assegnato proxy: {proxy.split('@')[1] if '@' in proxy else proxy}")
    
    session = init_session(proxy)
    ultimo_refresh = time.time()
    
    log(f"📧 Account: {account_name}")
    
    errori_consecutivi = 0
    MAX_ERRORI = 5
    captcha_counter = 0
    
    while True:
        if time.time() - ultimo_refresh > REFRESH_INTERVAL:
            log(f"[{account_name}] 🔄 Refresh periodico della sessione...")
            session = init_session(proxy)
            ultimo_refresh = time.time()
            errori_consecutivi = 0
        
        try:
            r = session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
                verify=False, timeout=REQUEST_TIMEOUT
            )
            
            if r.status_code != 200:
                errori_consecutivi += 1
                log(f"[{account_name}] ⚠️ HTTP {r.status_code}")
                if errori_consecutivi >= MAX_ERRORI:
                    log(f"[{account_name}] 🔄 Riavvio sessione per errore HTTP...")
                    session = init_session(proxy)
                    ultimo_refresh = time.time()
                    errori_consecutivi = 0
                time.sleep(5)
                continue
            
            data = r.json()
            
            # 🔍 CONTROLLA IL TIPO DI CAPTCHA
            picmap = data.get("picmap")
            surfses = data.get("surfses", {})
            urlid = surfses.get("urlid")
            qpic = surfses.get("qpic")
            seconds = int(surfses.get("seconds", 20))
            
            # 🔑 SE PICMAP È NULL → CAPTCHA MATEMATICO
            if picmap is None:
                log(f"[{account_name}] 🧮 Captcha matematico rilevato")
                
                # Scarica l'immagine
                try:
                    img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                except Exception as e:
                    log(f"[{account_name}] ❌ Errore scaricamento immagine: {e}")
                    salva_captcha_analyzer(supabase_client, account_name, qpic, None, None, None, "matematico_errore_scaricamento", urlid, stats)
                    # Continua con il prossimo captcha invece di fermare l'account
                    time.sleep(random.uniform(1.5, 3.0))
                    continue
                
                # Risolvi con il risolutore matematico
                segno, risultato, a, b = risolvi_captcha_matematico(img, surfses)
                
                if segno:
                    word = numero_a_parola(risultato)
                    log(f"[{account_name}] ✅ Risolto: {a} {segno} {b} = {risultato} → {word}")
                    
                    # Attendi il tempo minimo
                    time.sleep(seconds)
                    
                    # Invia risposta
                    url = f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2&ajax=1&word={word}&screen_width=1024&screen_height=768"
                    url += "&window_width=1024&window_height=643&top_width=1024&top_height=50"
                    url += "&fpcode=TW96aWxsYTsgTmV0c2NhcGU7IDUuMCAoV2luZG93cyk7IFdpbjMy"
                    url += f"&cit={int(time.time() * 1000)}&try=1"
                    
                    resp = session.get(url, verify=False, timeout=REQUEST_TIMEOUT)
                    response_data = resp.json()
                    
                    if response_data.get("warning") == "wrong_choice":
                        log(f"[{account_name}] ❌ Risposta sbagliata ({word}) - salvo")
                        # Salva quelli sbagliati
                        salva_captcha_analyzer(supabase_client, account_name, qpic, img, None, None, "matematico_wrong", urlid, stats)
                        # Continua con il prossimo captcha
                    else:
                        log(f"[{account_name}] ✅ OK!")
                        stats['risolti'] += 1
                else:
                    log(f"[{account_name}] ❌ Non risolto - salvo")
                    # Salva quelli non risolti
                    salva_captcha_analyzer(supabase_client, account_name, qpic, img, None, None, "matematico_non_risolto", urlid, stats)
                
                # Attendi prima del prossimo captcha (non fermare l'account)
                time.sleep(random.uniform(1.5, 3.0))
                continue
            
            # SE PICMAP ESISTE → CAPTCHA A FIGURE
            if not urlid or not qpic:
                errori_consecutivi += 1
                log(f"[{account_name}] ⚠️ urlid=None, qpic=None ({errori_consecutivi}/{MAX_ERRORI})")
                if errori_consecutivi >= MAX_ERRORI:
                    log(f"[{account_name}] 🔄 Riavvio sessione per captcha assente...")
                    session = init_session(proxy)
                    ultimo_refresh = time.time()
                    errori_consecutivi = 0
                time.sleep(5)
                continue
            
            errori_consecutivi = 0
            
            # Scarica l'immagine
            img_data = session.get(
                f"https://www.easyhits4u.com/simg/{qpic}.jpg",
                verify=False
            ).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Risolvi captcha a figure
            crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
            labels = []
            for crop in crops:
                if crop is not None and crop.size > 0:
                    label = predict_figure(crop)
                    labels.append(label)
                else:
                    labels.append(None)
            
            seen = {}
            chosen_idx = None
            for i, label in enumerate(labels):
                if label and label != "errore":
                    if label in seen:
                        chosen_idx = seen[label]
                        break
                    seen[label] = i
            
            if chosen_idx is None:
                log(f"[{account_name}] ❌ Nessun duplicato trovato")
                salva_captcha_analyzer(supabase_client, account_name, qpic, img, picmap, labels, "nessun_duplicato", urlid, stats)
                # Ferma l'account solo per i captcha figure
                return
            
            word = picmap[chosen_idx]["value"]
            
            # 🔑 ATTESA CON RITARDO CASUALE
            delay_extra = random.uniform(0.5, 3.0)
            total_delay = seconds + delay_extra
            log(f"[{account_name}] ⏳ Attesa {total_delay:.1f} secondi ({seconds}s + {delay_extra:.1f}s extra)...")
            time.sleep(total_delay)
            
            # Invia risposta
            url = f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2&ajax=1&word={word}&screen_width=1024&screen_height=768"
            url += "&window_width=1024&window_height=643&top_width=1024&top_height=50"
            url += "&fpcode=TW96aWxsYTsgTmV0c2NhcGU7IDUuMCAoV2luZG93cyk7IFdpbjMy"
            url += f"&cit={int(time.time() * 1000)}&try=1"
            
            resp = session.get(url, verify=False, timeout=REQUEST_TIMEOUT)
            response_data = resp.json()
            
            if response_data.get("warning") == "wrong_choice":
                log(f"[{account_name}] ❌ Risposta sbagliata: {word}")
                salva_captcha_analyzer(supabase_client, account_name, qpic, img, picmap, labels, "wrong_choice", urlid, stats)
                # Per i figure, ferma l'account
                return
            
            captcha_counter += 1
            stats['risolti'] += 1
            if captcha_counter % 10 == 0:
                log(f"[{account_name}] ✅ OK #{captcha_counter}")
            
            time.sleep(random.uniform(1.5, 3.0))
            
        except Exception as e:
            log(f"[{account_name}] ❌ Errore: {e}")
            errori_consecutivi += 1
            if errori_consecutivi >= MAX_ERRORI:
                log(f"[{account_name}] 🔄 Riavvio sessione per errore...")
                session = init_session(proxy)
                ultimo_refresh = time.time()
                errori_consecutivi = 0
            time.sleep(5)

# ==================== MAIN ====================
def main():
    log("=" * 60)
    log("🚀 COLLECTOR ANALYZER V3 - CON RISOLUTORE MATEMATICO")
    log("=" * 60)
    
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    log(f"📁 Captcha DB: {SUPABASE_URL}")
    
    cookie_supabase = create_client(COOKIE_SUPABASE_URL, COOKIE_SUPABASE_KEY)
    log(f"📁 Cookie DB: {COOKIE_SUPABASE_URL}")
    
    if not load_dataset_from_hf():
        log("❌ Dataset non caricato")
        return
    
    # Inizializza OCR matematici
    if not init_math_ocr():
        log("⚠️ OCR matematici non inizializzati, continuo solo con le figure")
    
    try:
        result = cookie_supabase.table('account_cookies')\
            .select('account_name, cookie_string')\
            .in_('account_name', NUOVI_ACCOUNT)\
            .execute()
        
        cookies = {}
        for row in result.data:
            if row.get('cookie_string'):
                cookies[row['account_name']] = row['cookie_string']
        
        log(f"📋 Letti {len(cookies)} cookie dei {len(NUOVI_ACCOUNT)} nuovi account")
    except Exception as e:
        log(f"❌ Errore lettura cookie: {e}")
        return
    
    if not cookies:
        log("❌ Nessun cookie trovato per i 40 nuovi account")
        return
    
    stats = {'figure': 0, 'math': 0, 'errors': 0, 'surf': 0, 'risolti': 0}
    
    threads = []
    for account_name, cookie_string in cookies.items():
        while len(threads) >= MAX_CONCURRENT:
            threads = [t for t in threads if t.is_alive()]
            time.sleep(1)
        
        t = threading.Thread(
            target=surf_account,
            args=(account_name, cookie_string, stats, supabase_client)
        )
        t.start()
        threads.append(t)
        time.sleep(STAGGERED_START_DELAY)
    
    for t in threads:
        t.join()
    
    log("=" * 60)
    log("📊 STATISTICHE FINALI")
    log(f"   Figure salvate: {stats['figure']}")
    log(f"   Matematici salvati: {stats['math']}")
    log(f"   Captcha risolti: {stats['risolti']}")
    log("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n🛑 Interrotto")
        sys.exit(0)
