import time

import h5py
from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

# HDF5ファイルからデータを読み込む
with h5py.File("data/synthetic_medical_video.h5", "r") as f:
    # data/synthetic_medical_video.h5ファイルを存在し、/synthetic_videoデータコレクションも含めている確保
    data = f["/synthetic_video"][()]

# 単一チャネル動画データの抽出
# データ形式を[フレーム数、高さ、幅、チャネル数]と仮定し、全フレームの第1チャネル情報を保持
video_data = data[:, :, :, 0]

@app.route('/')
def index():
    return render_template("index.html")

"""
    再生データ作成：
        リアルタイム動画ストリームの生成・伝送
    引数：
        なし
    戻り値：
        なし
"""
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            for frame in video_data:
                # BGR画像に変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # BGR画像JPEGに変換
                _, buffer = cv2.imencode('.jpg', frame_rgb)
                """
                    ジェネレータ関数、リアルタイム動画のフレームを作成するHTTP multipart/x-mixed-replaceリアクティビティーチャンク
                引数：
                    buffer：画像データを含めてbufferオブジェクト、メソッドtobytes()をサポート必要
                戻り値：
                    bytes：HTTPチャンク転送仕様に準拠したJPEG画像フレームのバイト列、下記の内容を含む
                        - フレーム境界・開始（--frame）
                        - 内容タイプのタイトル (Content-Type: image/jpeg)
                        - 区切（\r\n）
                        - 画像データ
                        - フレーム境界 ・終了(\r\n)
                """
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                # フレーム間に50msの遅延を追加
                time.sleep(0.05)

    # 作成したフレーム返す
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)