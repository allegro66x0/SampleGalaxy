import sys
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider, 
                           QHBoxLayout, QLabel, QCheckBox, QScrollArea, QGroupBox, QPushButton)
from PyQt6.QtCore import Qt, QUrl, QMimeData, QPoint, QPointF
from PyQt6.QtGui import QDrag, QColor, QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
import pyqtgraph as pg
import numpy as np

class GalaxyPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('k')
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setMouseEnabled(x=True, y=True)
        self.setMenuEnabled(False) # 右クリックメニューを無効化
        
        # 軌跡描画用 (Trail)
        self.trail_curve = pg.PlotCurveItem(
            pen=pg.mkPen('w', width=2, style=Qt.PenStyle.SolidLine),
            clickable=False
        )
        self.trail_curve.setZValue(15) # 散布図(10)より上、マーカー(20)より下
        self.addItem(self.trail_curve)

        # 現在の選択マーカー (Selection Marker)
        self.selection_marker = pg.ScatterPlotItem(
            size=20,
            pen=pg.mkPen('cyan', width=3),
            brush=pg.mkBrush(None),
            symbol='o',
            pxMode=True
        )
        self.selection_marker.setZValue(20) # 最前面
        # マウスイベントを無視して、下の点がクリックできるようにする
        self.selection_marker.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.addItem(self.selection_marker)

        # 散布図アイテム (Scatter Plot Item)
        # hoverable=Trueにするとポイントが多い時に重くなる可能性があるため注意
        self.scatter = pg.ScatterPlotItem(
            size=10, 
            pen=pg.mkPen(None), 
            brush=pg.mkBrush(255, 255, 255, 120),
            hoverable=True,
            hoverBrush=pg.mkBrush(255, 255, 255, 255),
            hoverSize=15
        )
        self.scatter.setZValue(10)
        self.addItem(self.scatter)
        
        # データ保存用
        self.points_data = []
        self.playback_history = [] # 履歴 (x, y) のリスト
        self.coords_array = None # 高速検索用Numpy配列
        self.current_pos = None # 現在再生中の座標
        
        # フィルター設定
        self.filter_oneshot = False
        self.visible_categories = set() # 表示するカテゴリのセット (空なら全て表示、または初期化時に設定)
        
        # 検索用キャッシュ (表示されている点のみ)
        self.visible_points_data = [] # 表示中のメタデータリスト
        self.visible_coords_cache = None # 表示中の座標Numpy配列 (N, 2)

        # カテゴリ別カラーパレット (初期化時に定義)
        self.category_colors = {
            "KICK": (220, 20, 60),      # Crimson (赤)
            "SNARE": (255, 140, 0),     # DarkOrange (オレンジ)
            "CLAP": (255, 69, 0),       # OrangeRed (朱色)
            "HIHAT": (0, 255, 255),     # Cyan (水色)
            "CRASH": (70, 130, 180),    # SteelBlue
            "TOM": (139, 69, 19),       # SaddleBrown (茶色)
            "BASS": (138, 43, 226),     # BlueViolet (紫)
            "GUITAR": (255, 215, 0),    # Gold (黄色)
            "PIANO": (50, 205, 50),     # LimeGreen (緑)
            "LOOP": (0, 100, 0),        # DarkGreen (深緑)
            "FX": (30, 144, 255),       # DodgerBlue (青)
            "UNKNOWN": (128, 128, 128)  # Gray
        }
        # 初期状態では全カテゴリを表示
        self.visible_categories = set(self.category_colors.keys())
        # 互換性のため追加キーも含める
        self.visible_categories.add("TOP_LOOP")
        self.visible_categories.add("PERC")
        self.visible_categories.add("OTHER")

        # オーディオプレイヤー
        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        
        # クリックイベントの接続 (Scene全体)
        # 散布図アイテムのクリック(sigClicked)は範囲が狭いため、
        # Sceneのクリックイベントを拾って最近傍探索を行う (Smart Click)
        self.scene().sigMouseClicked.connect(self.on_scene_clicked)
        # self.scatter.sigClicked.connect(self.on_point_clicked) # 旧方式は無効化
        
        # ドラッグ＆ドロップの状態管理
        self.drag_start_pos = None
        self.current_hovered_point = None
        self.last_played_path = None # スクラビング再生の重複防止用
        
        # キー入力のためのフォーカス設定
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def load_data(self, json_path):
        if not os.path.exists(json_path):
            print(f"データベースが見つかりません: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.points_data = data
        
        # 全データの座標配列 (Jump機能などで使用)
        self.coords_array = np.array([[item['x'], item['y']] for item in data])
        
        self.update_plot()
        
    def update_plot(self):
        self.scatter.clear()
        spots = []
        
        # キャッシュのリセット
        self.visible_points_data = []
        visible_coords_list = []
        
        # フォールバック用パレット
        cluster_palette = [
            (100, 100, 100), (150, 150, 150), (200, 200, 200)
        ]
        
        # 旧カテゴリ名との互換性
        colors = self.category_colors.copy()
        colors.update({
            "TOP_LOOP": (0, 128, 0),
            "PERC": (255, 105, 180),
            "OTHER": (180, 180, 180)
        })
        
        for item in self.points_data:
            # 1. ワンショットフィルター適用
            if self.filter_oneshot:
                duration = item.get('duration', 0.0)
                if duration > 2.0:
                    continue

            # 2. カテゴリフィルター適用
            category = item.get('category', 'UNKNOWN')
            if category not in self.visible_categories:
                # 未定義のカテゴリの場合、UNKNOWNとして扱うかチェック
                if 'UNKNOWN' in self.visible_categories and category not in colors:
                    pass # UNKNOWNが表示ONなら表示
                else: 
                     continue # 表示OFF

            # 色の決定
            if category in colors:
                color = colors[category]
            else:
                cluster_id = item.get('cluster', 0)
                color = cluster_palette[cluster_id % len(cluster_palette)]
            
            # ユーザー要望: 不透明にする、サイズを小さくする
            size = 4
            alpha = 255
            
            spots.append({
                'pos': (item['x'], item['y']),
                'data': item['path'],
                'brush': pg.mkBrush(*color, alpha),
                'symbol': 'o',
                'size': size
            })
            
            # 検索用キャッシュに追加
            self.visible_points_data.append(item)
            visible_coords_list.append([item['x'], item['y']])
            
        self.scatter.addPoints(spots)
        
        # Numpy配列化
        if visible_coords_list:
            self.visible_coords_cache = np.array(visible_coords_list)
        else:
            self.visible_coords_cache = None

    def find_nearest_point(self, scene_pos, threshold=20):
        """
        指定されたシーン座標に最も近い点を探す
        threshold: 判定距離 (ピクセル単位)
        戻り値: (file_path, data_pos, distance) or None
        """
        if self.visible_coords_cache is None or len(self.visible_coords_cache) == 0:
            return None

        # 1. マウス位置 (Scene) をデータ座標 (View) に変換
        mouse_point = self.plotItem.vb.mapSceneToView(scene_pos)
        mouse_arr = np.array([mouse_point.x(), mouse_point.y()])
        
        # 2. データ空間での最近傍探索 (高速)
        dists = np.linalg.norm(self.visible_coords_cache - mouse_arr, axis=1)
        min_idx = np.argmin(dists)
        
        # 3. 画面上の距離 (Pixel) で判定
        nearest_data_pos = QPointF(self.visible_coords_cache[min_idx][0], self.visible_coords_cache[min_idx][1])
        nearest_scene_pos = self.plotItem.vb.mapViewToScene(nearest_data_pos)
        
        pixel_dist = (nearest_scene_pos - scene_pos).manhattanLength()
        
        if pixel_dist < threshold:
            target_item = self.visible_points_data[min_idx]
            return target_item['path'], nearest_data_pos, pixel_dist
            
        return None

    def on_scene_clicked(self, event):
        """
        スマートクリック判定 (Nearest Neighbor Search)
        """
        if event.button() != Qt.MouseButton.LeftButton:
            return
            
        scene_pos = event.scenePos()
        result = self.find_nearest_point(scene_pos)
        
        if result:
            file_path, data_pos, _ = result
            
            # 再生済み更新
            self.last_played_path = file_path

            # 履歴と表示を更新
            self.update_history(data_pos)
            
            # データからstart_timeを取得 (キャッシュから)
            # visible_points_dataの中から探しても良いが、パスで特定済み
            start_time = 0
            # 高速化のために points_data を走査するのは避ける (visible_points_dataにあるはず)
            # find_nearest_point で target_item を返せばいいが、リファクタリングの影響範囲を抑える
            # ここではシンプルに再検索 (許容範囲)
            target_item = next((item for item in self.visible_points_data if item['path'] == file_path), None)
            if target_item:
                start_time = target_item.get('start_time', 0)
                
            self.play_audio(file_path, start_time)

    def update_history(self, pos):
        x, y = pos.x(), pos.y()
        self.current_pos = (x, y)
        
        self.playback_history.append((x, y))
        if len(self.playback_history) > 5:
            self.playback_history.pop(0)
            
        xs = [p[0] for p in self.playback_history]
        ys = [p[1] for p in self.playback_history]
        self.trail_curve.setData(xs, ys)
        
        self.selection_marker.setData([x], [y])

    def jump_to_random_neighbor(self):
        if self.current_pos is None or self.coords_array is None:
            return
            
        curr = np.array(self.current_pos)
        dists = np.linalg.norm(self.coords_array - curr, axis=1)
        
        nearest_indices = np.argsort(dists)[1:51]
        
        if len(nearest_indices) > 0:
            target_idx = np.random.choice(nearest_indices)
            target_item = self.points_data[target_idx]
            
            new_pos_f = QPointF(target_item['x'], target_item['y'])
            
            self.update_history(new_pos_f)
            self.play_audio(target_item['path'], target_item.get('start_time', 0))
            print(f"Jumped to neighbor: {target_item['path']}")

    def play_audio(self, file_path, start_time=0):
        url = QUrl.fromLocalFile(os.path.abspath(file_path))
        self.player.stop()
        self.player.setSource(url)
        
        start_ms = int(start_time * 1000)
        if start_ms > 0:
            self.player.setPosition(start_ms)
            
        self.player.play()
        print(f"再生中: {file_path} (Start: {start_time:.2f}s)")
        
    def set_filter_oneshot(self, enabled):
        self.filter_oneshot = enabled
        self.update_plot()
        
    def set_category_visibility(self, category, visible):
        if visible:
            self.visible_categories.add(category)
        else:
            self.visible_categories.discard(category)
        self.update_plot()
        
    def set_all_categories_visibility(self, visible):
        if visible:
            self.visible_categories = set(self.category_colors.keys())
            self.visible_categories.add("TOP_LOOP")
            self.visible_categories.add("PERC")
            self.visible_categories.add("OTHER")
        else:
            self.visible_categories.clear()
        self.update_plot()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self.jump_to_random_neighbor()
        else:
            super().keyPressEvent(ev)

    def mousePressEvent(self, ev):
        # フォーカスを取得してキー入力を受け取れるようにする
        self.setFocus()
        
        if ev.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = ev.pos()
        elif ev.button() == Qt.MouseButton.RightButton:
            # 右クリックはスクラビングの開始だが、クリック時点でも再生したい
            # ただし、mouseMoveEventで処理されるのでここではacceptのみでも良いが、
            # 即座に音が出てほしい場合はここで判定してもよい。
            # 今回は「動きながら」がメインだが、開始点でも鳴らす
            try:
                scene_pos = self.mapToScene(ev.pos())
                result = self.find_nearest_point(scene_pos, threshold=30)
                if result:
                    file_path, data_pos, _ = result
                    self.last_played_path = file_path
                    self.update_history(data_pos)
                    target_item = next((item for item in self.visible_points_data if item['path'] == file_path), None)
                    if target_item:
                        self.play_audio(file_path, target_item.get('start_time', 0))
            except Exception as e:
                print(f"Right click error: {e}")
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        # 右ドラッグ（スクラビング再生）
        if ev.buttons() & Qt.MouseButton.RightButton:
            try:
                scene_pos = self.mapToScene(ev.pos())
                result = self.find_nearest_point(scene_pos, threshold=30) # 少し広め
                
                if result:
                    file_path, data_pos, _ = result
                    
                    # 連続再生防止 (前回と同じなら再生しない)
                    if file_path != self.last_played_path:
                        self.last_played_path = file_path
                        self.update_history(data_pos)
                        
                        target_item = next((item for item in self.visible_points_data if item['path'] == file_path), None)
                        if target_item:
                            self.play_audio(file_path, target_item.get('start_time', 0))
            except Exception as e:
                print(f"Scrubbing error: {e}")
            return

        # 左ドラッグ（DAWへのドロップ）
        if not (ev.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(ev)
            return
        if not self.drag_start_pos:
            return

        dist = (ev.pos() - self.drag_start_pos).manhattanLength()
        if dist < QApplication.startDragDistance():
            return
            
        try:
            # ドラッグ開始位置周辺の点を探す
            scene_pos = self.mapToScene(self.drag_start_pos)
            # 判定範囲を少し広めにする (ドラッグのしやすさ重視)
            result = self.find_nearest_point(scene_pos, threshold=30)
            
            if result:
                file_path, _, _ = result
                self.start_drag(file_path)
                
        except Exception as e:
            print(f"Drag detection error: {e}")
        
        super().mouseMoveEvent(ev)

    def start_drag(self, file_path):
        drag = QDrag(self)
        mime_data = QMimeData()
        abs_path = os.path.abspath(file_path)
        url = QUrl.fromLocalFile(abs_path)
        mime_data.setUrls([url])
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.CopyAction)

    def get_player(self):
        return self.player

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sample Galaxy")
        self.resize(1200, 800) # 横幅を少し広げる
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main Layout (Horizontal: Sidebar + Plot)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Sidebar (Filters) ---
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(200) # 固定幅
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. 表示設定グループ
        disp_group = QGroupBox("Display Options")
        disp_layout = QVBoxLayout()
        
        # One-Shot Filter
        self.one_shot_checkbox = QCheckBox("One-Shot (< 2s)")
        self.one_shot_checkbox.stateChanged.connect(self.on_oneshot_changed)
        disp_layout.addWidget(self.one_shot_checkbox)
        
        disp_group.setLayout(disp_layout)
        self.sidebar_layout.addWidget(disp_group)
        
        # 2. カテゴリフィルターグループ (Scrollable if needed)
        cat_group = QGroupBox("Tags (Categories)")
        cat_layout = QVBoxLayout()
        
        # 一括操作ボタン
        btn_layout = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_all.clicked.connect(self.on_select_all)
        btn_none.clicked.connect(self.on_deselect_all)
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_none)
        cat_layout.addLayout(btn_layout)
        
        # Scroll Area for tags
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # プロットウィジェットを先に作ってカテゴリ情報を取得
        self.plot_widget = GalaxyPlotWidget()
        
        # カテゴリチェックボックスの生成
        self.category_checkboxes = {}
        for cat, color in self.plot_widget.category_colors.items():
            # 色付きのテキストにするためのHTML
            color_hex = QColor(*color).name()
            # ラベル (e.g. ■ KICK)
            cb = QCheckBox(cat)
            cb.setChecked(True) # デフォルトON
            # スタイルシートで色をつけるか、ラベルをつけるか
            cb.setStyleSheet(f"QCheckBox {{ color: {color_hex}; font-weight: bold; }}")
            
            # クロージャで変数をキャプチャするために lambda cat=cat: ... を使用
            cb.stateChanged.connect(lambda state, c=cat: self.on_category_changed(c, state))
            
            scroll_layout.addWidget(cb)
            self.category_checkboxes[cat] = cb
            
        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        
        cat_layout.addWidget(scroll)
        cat_group.setLayout(cat_layout)
        
        self.sidebar_layout.addWidget(cat_group)
        
        # サイドバーをレイアウトに追加
        self.main_layout.addWidget(self.sidebar)
        
        # --- Right Side (Plot + Seek Bar) ---
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.right_layout.addWidget(self.plot_widget)
        
        # Seek Bar Area
        self.seek_layout = QHBoxLayout()
        self.seek_layout.setContentsMargins(10, 0, 10, 10)
        
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_layout.addWidget(self.seek_slider)
        
        self.right_layout.addLayout(self.seek_layout)
        
        self.main_layout.addWidget(self.right_widget)
        
        # プレイヤーのシグナル接続
        self.player = self.plot_widget.get_player()
        self.player.positionChanged.connect(self.update_slider)
        self.player.durationChanged.connect(self.update_duration)
        
        # スライダーの操作
        self.seek_slider.sliderMoved.connect(self.set_position)
        self.seek_slider.sliderPressed.connect(self.slider_pressed)
        self.seek_slider.sliderReleased.connect(self.slider_released)
        
        self.is_slider_pressed = False

        # データをロード
        self.plot_widget.load_data("database.json")

    def update_slider(self, position):
        if not self.is_slider_pressed:
            self.seek_slider.setValue(position)

    def update_duration(self, duration):
        self.seek_slider.setRange(0, duration)

    def set_position(self, position):
        self.player.setPosition(position)

    def slider_pressed(self):
        self.is_slider_pressed = True
        self.player.pause()

    def slider_released(self):
        self.is_slider_pressed = False
        self.set_position(self.seek_slider.value())
        self.player.play()

    def on_oneshot_changed(self):
        self.plot_widget.set_filter_oneshot(self.one_shot_checkbox.isChecked())
        
    def on_category_changed(self, category, state):
        is_visible = (state == Qt.CheckState.Checked.value or state == 2) # 2 is Checked
        self.plot_widget.set_category_visibility(category, is_visible)

    def on_select_all(self):
        # シグナルをブロックしてUI更新、その後一度だけプロット更新
        for cb in self.category_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self.plot_widget.set_all_categories_visibility(True)

    def on_deselect_all(self):
        for cb in self.category_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.plot_widget.set_all_categories_visibility(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
