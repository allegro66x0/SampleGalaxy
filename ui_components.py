import os
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QListWidget, QGraphicsSceneMouseEvent
from PyQt6.QtCore import Qt, QMimeData, QPointF, pyqtSignal, QUrl, QTimer
from PyQt6.QtGui import QDrag, QColor

from audio_engine import AudioEngine
import utils

class FavoritesListWidget(QListWidget):
    files_dropped = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
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
        self.setMenuEnabled(False)
        
        # Audio Engine
        self.audio_engine = AudioEngine()
        
        # --- Visual Setup ---
        # 1. Trail Curve
        self.trail_curve = pg.PlotCurveItem(
            pen=pg.mkPen('w', width=2, style=Qt.PenStyle.SolidLine),
            clickable=False
        )
        self.trail_curve.setZValue(15)
        self.addItem(self.trail_curve)

        # 2. Selection Marker
        self.selection_marker = pg.ScatterPlotItem(
            size=20,
            pen=pg.mkPen('cyan', width=3),
            brush=pg.mkBrush(None),
            symbol='o',
            pxMode=True
        )
        self.selection_marker.setZValue(20)
        self.selection_marker.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.addItem(self.selection_marker)

        # 3. Main Scatter Plot
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
        
        # --- Data & State ---
        self.points_data = []
        self.playback_history = []
        self.coords_array = None
        self.current_pos = None
        
        self.filter_oneshot = False
        self.filter_favorites_only = False
        
        self.favorites = set()
        self.favorites_file = "favorites.json"
        self.load_favorites()
        
        self.visible_points_data = []
        self.visible_coords_cache = None

        self.category_colors = {
            "KICK": (220, 20, 60),      # Crimson
            "SNARE": (255, 140, 0),     # DarkOrange
            "CLAP": (255, 69, 0),       # OrangeRed
            "HIHAT": (0, 255, 255),     # Cyan
            "CRASH": (70, 130, 180),    # SteelBlue
            "TOM": (139, 69, 19),       # SaddleBrown
            "BASS": (138, 43, 226),     # BlueViolet
            "GUITAR": (255, 215, 0),    # Gold
            "PIANO": (50, 205, 50),     # LimeGreen
            "LOOP": (0, 100, 0),        # DarkGreen
            "FX": (30, 144, 255),       # DodgerBlue
            "UNKNOWN": (128, 128, 128)  # Gray
        }
        self.visible_categories = set(self.category_colors.keys())
        self.visible_categories.add("TOP_LOOP")
        self.visible_categories.add("PERC")
        self.visible_categories.add("OTHER")

        # --- Events ---
        self.scene().sigMouseClicked.connect(self.on_scene_clicked)
        
        self.getPlotItem().hideAxis('bottom')
        self.getPlotItem().hideAxis('left')
        self.getPlotItem().getViewBox().setAspectLocked(True)
        
        self.drag_start_pos = None
        self.last_played_path = None
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # --- Favorites ---
    def load_favorites(self):
        data = utils.load_json(self.favorites_file)
        if data:
            self.favorites = {utils.normalize_path(p) for p in data}
        else:
            self.favorites = set()

    def save_favorites(self):
        utils.save_json(self.favorites_file, list(self.favorites))

    def add_favorite(self, path):
        path = utils.normalize_path(path)
        if path not in self.favorites:
            self.favorites.add(path)
            print(f"Added to favorites: {path}")
            self.save_favorites()
            self.update_plot()
            self.favorites_changed.emit()

    def remove_favorite(self, path):
        path = utils.normalize_path(path)
        if path in self.favorites:
            self.favorites.remove(path)
            print(f"Removed from favorites: {path}")
            self.save_favorites()
            self.update_plot()
            self.favorites_changed.emit()
            
    def toggle_favorite(self):
        if not self.last_played_path:
            return
        path = self.last_played_path
        if path in self.favorites:
            self.remove_favorite(path)
        else:
            self.add_favorite(path)
            
    # --- Filters ---
    def set_filter_favorites(self, enabled):
        self.filter_favorites_only = enabled
        self.update_plot()
        
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

    # --- Data Loading ---
    def load_data(self, json_path):
        data = utils.load_json(json_path)
        if not data:
            return
            
        self.points_data = data
        for item in self.points_data:
            if 'path' in item:
                item['path'] = utils.normalize_path(item['path'])
        
        self.coords_array = np.array([[item['x'], item['y']] for item in data])
        
        centroids = [item.get('centroid', 0) for item in data if 'centroid' in item]
        if centroids:
            self.min_centroid = min(centroids)
            self.max_centroid = max(centroids)
        else:
            self.min_centroid = 0
            self.max_centroid = 1
            
        if self.max_centroid == self.min_centroid:
            self.max_centroid += 1

        self.update_plot()
        
    def update_plot(self):
        self.scatter.clear()
        spots = []
        self.visible_points_data = []
        visible_coords_list = []
        
        cluster_palette = [(100, 100, 100), (150, 150, 150), (200, 200, 200)]
        colors = self.category_colors.copy()
        colors.update({"TOP_LOOP": (0, 128, 0), "PERC": (255, 105, 180), "OTHER": (180, 180, 180)})
        
        for item in self.points_data:
            path = item['path']
            is_fav = path in self.favorites
            
            if self.filter_favorites_only and not is_fav: continue
            if self.filter_oneshot and item.get('duration', 0.0) > 2.0: continue

            category = item.get('category', 'UNKNOWN')
            if category not in self.visible_categories:
                if 'UNKNOWN' not in self.visible_categories or category in colors:
                     continue 

            color = (150, 150, 150)
            size = 4
            symbol = 'o'
            
            if is_fav:
                color = (255, 215, 0)
                size = 7
                symbol = 'star'
            else:
                if category == "LOOP" and 'centroid' in item:
                    val = item['centroid']
                    norm = (val - self.min_centroid) / (self.max_centroid - self.min_centroid)
                    norm = max(0.0, min(1.0, norm))
                    r = int(0 + norm * 180)
                    g = int(80 + norm * 175)
                    b = int(0 + norm * 50)
                    color = (r, g, b)
                elif category in colors:
                    color = colors[category]
                else:
                    color = cluster_palette[item.get('cluster', 0) % len(cluster_palette)]
            
            spots.append({'pos': (item['x'], item['y']), 'data': item['path'], 'brush': pg.mkBrush(*color, 255), 'symbol': symbol, 'size': size})
            self.visible_points_data.append(item)
            visible_coords_list.append([item['x'], item['y']])
            
        self.scatter.addPoints(spots)
        if visible_coords_list:
            self.visible_coords_cache = np.array(visible_coords_list)
        else:
            self.visible_coords_cache = None

    # --- Interaction ---
    def find_nearest_point(self, scene_pos, threshold=20):
        if self.visible_coords_cache is None or len(self.visible_coords_cache) == 0:
            return None
        mouse_point = self.plotItem.vb.mapSceneToView(scene_pos)
        mouse_arr = np.array([mouse_point.x(), mouse_point.y()])
        dists = np.linalg.norm(self.visible_coords_cache - mouse_arr, axis=1)
        min_idx = np.argmin(dists)
        nearest_data_pos = QPointF(self.visible_coords_cache[min_idx][0], self.visible_coords_cache[min_idx][1])
        nearest_scene_pos = self.plotItem.vb.mapViewToScene(nearest_data_pos)
        
        if (nearest_scene_pos - scene_pos).manhattanLength() < threshold:
            return self.visible_points_data[min_idx]['path'], nearest_data_pos, 0
        return None

    def on_scene_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton: return
        result = self.find_nearest_point(event.scenePos())
        if result:
            path, data_pos, _ = result
            self.select_point(path)

    def select_point(self, path):
        path = utils.normalize_path(path)
        target_item = next((d for d in self.points_data if d['path'] == path), None)
        if target_item:
            self.last_played_path = path
            self.update_history(QPointF(target_item['x'], target_item['y']))
            self.audio_engine.play(path, target_item.get('start_time', 0))
        else:
            print(f"File not found: {path}")

    def update_history(self, pos):
        x, y = pos.x(), pos.y()
        self.current_pos = (x, y)
        self.playback_history.append((x, y))
        if len(self.playback_history) > 5: self.playback_history.pop(0)
        self.trail_curve.setData([p[0] for p in self.playback_history], [p[1] for p in self.playback_history])
        self.selection_marker.setData([x], [y])

    def jump_to_random_neighbor(self):
        if self.current_pos is None or self.coords_array is None: return
        curr = np.array(self.current_pos)
        dists = np.linalg.norm(self.coords_array - curr, axis=1)
        nearest_indices = np.argsort(dists)[1:51]
        if len(nearest_indices) > 0:
            target_idx = np.random.choice(nearest_indices)
            item = self.points_data[target_idx]
            self.update_history(QPointF(item['x'], item['y']))
            self.audio_engine.play(item['path'], item.get('start_time', 0))

    def get_player(self):
        return self.audio_engine.get_main_player()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self.jump_to_random_neighbor()
        elif ev.key() == Qt.Key.Key_F:
            self.toggle_favorite()
        else:
            super().keyPressEvent(ev)

    def mousePressEvent(self, ev):
        self.setFocus()
        if ev.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = ev.pos()
        elif ev.button() == Qt.MouseButton.RightButton:
            try:
                result = self.find_nearest_point(self.mapToScene(ev.pos()), threshold=30)
                if result:
                    path, data_pos, _ = result
                    self.last_played_path = path
                    self.update_history(data_pos)
                    target_item = next((i for i in self.visible_points_data if i['path'] == path), None)
                    if target_item:
                        self.audio_engine.play(path, target_item.get('start_time', 0), polyphonic=True)
            except Exception as e:
                print(f"Right click error: {e}")
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self.audio_engine.stop_all_scrubbing()
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.RightButton:
            try:
                result = self.find_nearest_point(self.mapToScene(ev.pos()), threshold=30) 
                if result:
                    path, data_pos, _ = result
                    if path != self.last_played_path:
                        self.last_played_path = path
                        self.update_history(data_pos)
                        target_item = next((i for i in self.visible_points_data if i['path'] == path), None)
                        if target_item:
                            self.audio_engine.play(path, target_item.get('start_time', 0), polyphonic=True)
            except Exception as e:
                print(f"Scrubbing error: {e}")
            return

        if not (ev.buttons() & Qt.MouseButton.LeftButton) or not self.drag_start_pos:
            super().mouseMoveEvent(ev)
            return
        if (ev.pos() - self.drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return
            
        try:
            result = self.find_nearest_point(self.mapToScene(self.drag_start_pos), threshold=30)
            if result:
                self.start_drag(result[0])
        except Exception: pass
        super().mouseMoveEvent(ev)

    def start_drag(self, file_path):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(os.path.abspath(file_path))])
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)
