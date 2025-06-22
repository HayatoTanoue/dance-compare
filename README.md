# Dance Compare

ダンスの練習が一人でできるようなインターフェースと解析機能を持ったアプリです。

## 機能

### 解析機能
- アップロードされた動画に対して骨格推定を行う

### GUI
- 教師動画のアップロード
- 生徒動画のアップロード
- 動画ビューワー
- 骨格重畳表示機能
- 秒数合わせ機能（異なる時間で撮影されている可能性があるので両方の再生スタート時刻合わせ）

## セットアップ

```bash
pip install -r requirements.txt
```

## 実行

```bash
streamlit run app.py
```

## 必要なライブラリ

- streamlit: GUI
- opencv-python: 動画処理
- mediapipe: 骨格推定
- numpy: 数値計算
- pillow: 画像処理
