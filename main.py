import sys
import json
import os
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider, 
                           QHBoxLayout, QLabel, QCheckBox, QScrollArea, QGroupBox, QPushButton,
                           QListWidget, QAbstractItemView, QListWidgetItem)
from PyQt6.QtCore import Qt, QUrl, QMimeData, QPoint, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import QDrag, QColor, QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
import pyqtgraph as pg
import numpy as np

class FavoritesListWidget(QListWidget):
    files_dropped = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(False) # リストからのドラッグは今回は不要（並べ替えしないなら）
        self.setDropIndicatorShown(True)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            files = [u.toLocalFile() for u in event.mimeData().urls()]
            self.files_dropped.emit(files)
            event.accept()

class GalaxyPlotWidget(pg.PlotWidget):
    favorites_changed = pyqtSignal()
    
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
        self.filter_favorites_only = False # お気に入り専用レイヤー
        self.visible_categories = set() # 表示するカテゴリのセット (空なら全て表示、または初期化時に設定)
        
        # お気に入りデータ
        self.favorites = set()
        self.favorites_file = "favorites.json"
        self.load_favorites()
        
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
        
        # Polyphonic Scrubbing Pool (for Right-Click Drag)
        self.scrub_pool = []
        self.scrub_pool_outputs = []
        self.pool_size = 8
        self.pool_index = 0
        
        for _ in range(self.pool_size):
            p = QMediaPlayer()
            o = QAudioOutput()
            p.setAudioOutput(o)
            o.setVolume(0.8)
            self.scrub_pool.append(p)
            self.scrub_pool_outputs.append(o)
            
        # スクラビング停止用タイマー (Watchdog)
        self.scrub_stop_timer = QTimer(self)
        self.scrub_stop_timer.setSingleShot(True)
        self.scrub_stop_timer.timeout.connect(self.stop_all_scrubbing)
        
        # クリックイベントの接続 (Scene全体)
        # 散布図アイテムのクリック(sigClicked)は範囲が狭いため、
        # Sceneのクリックイベントを拾って最近傍探索を行う (Smart Click)
        self.scene().sigMouseClicked.connect(self.on_scene_clicked)
        # self.scatter.sigClicked.connect(self.on_point_clicked) # 旧方式は無効化
        
        # ドラッグ＆ドロップの状態管理
        self.drag_start_pos = None
        self.current_hovered_point = None
        self.last_played_path = None # スクラビング再生の重複防止用
        self.last_play_time = 0 # スクラビング再生の間引き用 (0.1秒間隔)
        
        # キー入力のためのフォーカス設定
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def load_favorites(self):
        if os.path.exists(self.favorites_file):
            try:
                with open(self.favorites_file, 'r', encoding='utf-8') as f:
                    self.favorites = set(json.load(f))
            except Exception as e:
                print(f"Failed to load favorites: {e}")
                self.favorites = set()

    def save_favorites(self):
        try:
            with open(self.favorites_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.favorites), f, indent=4)
        except Exception as e:
            print(f"Failed to save favorites: {e}")

    def add_favorite(self, path):
        if path not in self.favorites:
            self.favorites.add(path)
            print(f"Added to favorites: {path}")
            self.save_favorites()
            self.update_plot()
            self.favorites_changed.emit()

    def remove_favorite(self, path):
        if path in self.favorites:
            self.favorites.remove(path)
            print(f"Removed from favorites: {path}")
            self.save_favorites()
            self.update_plot()
            self.favorites_changed.emit()

    def stop_all_scrubbing(self):
        # スクラビング用のプレイヤーを全て停止
        # print("Stopping all scrubbing players (Timeout)")
        for p in self.scrub_pool:
            if p.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                p.stop()

    def toggle_favorite(self):
        if not self.last_played_path:
            return
            
        path = self.last_played_path
        if path in self.favorites:
            self.remove_favorite(path)
        else:
            self.add_favorite(path)
            
    def set_filter_favorites(self, enabled):
        self.filter_favorites_only = enabled
        self.update_plot()
            
    def load_data(self, json_path):
        if not os.path.exists(json_path):
            print(f"データベースが見つかりません: {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.points_data = data
        
        # 全データの座標配列 (Jump機能などで使用)
        self.coords_array = np.array([[item['x'], item['y']] for item in data])
        
        # Centroidの範囲を計算 (グラデーション用)
        centroids = [item.get('centroid', 0) for item in data if 'centroid' in item]
        if centroids:
            self.min_centroid = min(centroids)
            self.max_centroid = max(centroids)
        else:
            self.min_centroid = 0
            self.max_centroid = 1
            
        # 外れ値の影響を抑えるために、少し範囲を狭める（上下5%カットなどを簡易的にやる）
        # ここでは簡易的に上下限をそのまま使う
        if self.max_centroid == self.min_centroid:
            self.max_centroid += 1

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
            path = item['path']
            is_fav = path in self.favorites
            
            # 0. お気に入りフィルター (AND検索ロジック)
            if self.filter_favorites_only and not is_fav:
                continue

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

            # 色とサイズの決定
            color = (150, 150, 150) # default
            size = 4
            symbol = 'o'
            
            if is_fav:
                # お気に入りは金色で強調
                if self.filter_favorites_only:
                     # フィルター中は元の色を使うか？いや、やはり金色で統一感がいいか。
                     # カテゴリ色も見たい場合は、枠線を金にする手もあるが、PyQtGraphのScatterはシンプルに
                     color = (255, 215, 0) # Gold
                else:
                     color = (255, 215, 0) # Gold
                size = 7 # 少し大きく
                symbol = 'star' # 星形 (pyqtgraphでサポートされている場合)
            else:
                if category == "LOOP" and 'centroid' in item:
                    # グラデーション計算
                    # 緑(0, 80, 0) -> 黄緑(180, 255, 50)
                    val = item['centroid']
                    norm = (val - self.min_centroid) / (self.max_centroid - self.min_centroid)
                    norm = max(0.0, min(1.0, norm)) # Clamp
                    
                    # Dark Green (Low) -> Bright Yellow-Green (High)
                    r = int(0 + norm * 180)
                    g = int(80 + norm * 175)
                    b = int(0 + norm * 50)
                    color = (r, g, b)
                    
                elif category in colors:
                    color = colors[category]
                else:
                    cluster_id = item.get('cluster', 0)
                    color = cluster_palette[cluster_id % len(cluster_palette)]
            
            # ユーザー要望: 不透明にする、サイズを小さくする
            alpha = 255
            
            spots.append({
                'pos': (item['x'], item['y']),
                'data': item['path'],
                'brush': pg.mkBrush(*color, alpha),
                'symbol': symbol,
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

    def play_audio(self, file_path, start_time=0, polyphonic=False):
        url = QUrl.fromLocalFile(os.path.abspath(file_path))
        
        if polyphonic:
            # Sound Poolからプレイヤーを選択 (Round Robin)
            player = self.scrub_pool[self.pool_index]
            self.pool_index = (self.pool_index + 1) % self.pool_size
            
            # 既存の再生を止めずに、上書き再生 (前の音が残るわけではないが、別のPlayerが担当するので前の音は消えない)
            if player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                 player.stop()
                 
            player.setSource(url)
            start_ms = int(start_time * 1000)
            if start_ms > 0:
                player.setPosition(start_ms)
            player.play()
            
        else:
            # Main Player (Single Voice, UI連携)
            self.player.stop()
            self.player.setSource(url)
            
            start_ms = int(start_time * 1000)
            if start_ms > 0:
                self.player.setPosition(start_ms)
                
            self.player.play()
            
        print(f"再生中: {file_path} (Start: {start_time:.2f}s, Poly: {polyphonic})")
        
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
        elif ev.key() == Qt.Key.Key_F:
            self.toggle_favorite()
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
                        # ヒット中はタイマーを止める（再生維持）
                        self.scrub_stop_timer.stop()
                        self.play_audio(file_path, target_item.get('start_time', 0), polyphonic=True)
            except Exception as e:
                print(f"Right click error: {e}")
            ev.accept()
            return
        super().mousePressEvent(ev)
        
    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            # 右クリック離したら即停止
            self.stop_all_scrubbing()
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        # 右ドラッグ（スクラビング再生）
        if ev.buttons() & Qt.MouseButton.RightButton:
            try:
                scene_pos = self.mapToScene(ev.pos())
                # 判定範囲を少し広めにする
                result = self.find_nearest_point(scene_pos, threshold=30) 
                
                if result:
                    # エリア内: タイマー停止（安全地帯）
                    self.scrub_stop_timer.stop()
                    
                    file_path, data_pos, _ = result
                    
                    # 連続再生防止 (前回と同じなら再生しない)
                    if file_path != self.last_played_path:
                        self.last_played_path = file_path
                        self.update_history(data_pos)
                        
                        target_item = next((item for item in self.visible_points_data if item['path'] == file_path), None)
                        if target_item:
                            self.play_audio(file_path, target_item.get('start_time', 0), polyphonic=True)
                else:
                    # エリア外: カウントダウン開始
                    # まだ停止タイマーが動いていなければ開始（動き続けている限りタイマーをリセットしない＝出た瞬間から0.4秒）
                    if not self.scrub_stop_timer.isActive():
                        self.scrub_stop_timer.start(400) # 0.4s
                        
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

        # Favorites Filter
        self.fav_checkbox = QCheckBox("Show Favorites Only (AND)")
        self.fav_checkbox.stateChanged.connect(self.on_fav_filter_changed)
        disp_layout.addWidget(self.fav_checkbox)
        
        disp_group.setLayout(disp_layout)
        self.sidebar_layout.addWidget(disp_group)
        
        # 1.5 Favorites List
        fav_list_group = QGroupBox("Favorites List")
        fav_list_layout = QVBoxLayout()
        
        self.fav_list_widget = FavoritesListWidget() # Custom Widget
        # self.fav_list_widget.setDragEnabled(True) # 無効化済み
        # self.fav_list_widget.setAcceptDrops(True) # クラス内で設定済み
        # self.fav_list_widget.setDropIndicatorShown(True)
        self.fav_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.fav_list_widget.itemClicked.connect(self.on_fav_item_clicked)
        # ドロップ信号の接続
        self.fav_list_widget.files_dropped.connect(self.on_files_dropped)
        
        fav_list_layout.addWidget(self.fav_list_widget)
        fav_list_group.setLayout(fav_list_layout)
        self.sidebar_layout.addWidget(fav_list_group)
        
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
        
        # お気に入り初期化
        self.plot_widget.favorites_changed.connect(self.update_favorites_list)
        self.update_favorites_list()

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

    def on_fav_filter_changed(self):
        self.plot_widget.set_filter_favorites(self.fav_checkbox.isChecked())
        
    def on_files_dropped(self, files):
        for path in files:
            # データベースにあるファイルかチェックしたほうが良いが、
            # とりあえず登録してしまう (表示時にポイントがなければ単にリストにあるだけになる)
            self.plot_widget.add_favorite(path)

    def update_favorites_list(self):
        self.fav_list_widget.clear()
        for path in sorted(self.plot_widget.favorites):
            name = os.path.basename(path)
            
            item = QListWidgetItem(self.fav_list_widget)
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            # item.setText(name) # カスタムウィジェットを使うのでテキストは不要だが、ソート等のためにあってもよい
            
            # カスタムウィジェット (Label + Remove Button)
            widget = QWidget()
            layout = QHBoxLayout()
            layout.setContentsMargins(5, 2, 5, 2)
            
            label = QLabel(name)
            # ラベルクリック透過させるのは難しいので、このウィジェット全体をリストアイテムとして扱う
            
            btn_remove = QPushButton("×")
            btn_remove.setFixedSize(24, 24)
            # lambdaでpathをキャプチャする際、変数が上書きされないようにデフォルト引数を使う
            btn_remove.clicked.connect(lambda checked, p=path: self.plot_widget.remove_favorite(p))
            
            layout.addWidget(label)
            layout.addStretch()
            layout.addWidget(btn_remove)
            widget.setLayout(layout)
            
            self.fav_list_widget.setItemWidget(item, widget)
            
    def on_fav_item_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        # 該当するファイルの座標を探してジャンプ＆再生
        target_item = next((d for d in self.plot_widget.points_data if d['path'] == path), None)
        if target_item:
            self.plot_widget.play_audio(path, target_item.get('start_time', 0))
            # 座標を中心に移動できればベストだが、今回はとりあえず再生のみ
            # 必要なら zoom を調整するコードなどを追加
        
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
