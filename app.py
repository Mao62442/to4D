import time

import h5py
from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

# HDF5ファイルからデータを読み込む
with h5py.File("data/synthetic_medical_video.h5", "r") as f:
    data = f["/synthetic_video"][()]  # shape: (T, H, W, C)

# 仮の4Dデータ（T=30フレーム、256x256、グレースケール）
video_data = data[:, :, :, 0]  # (T, H, W, C) → (T, H, W)に変換

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:  # フレームをループ再生
            for frame in video_data:
                # グレースケール画像を3チャンネルに変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                _, buffer = cv2.imencode('.jpg', frame_rgb)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)  # フレーム間に50msの遅延を追加

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)