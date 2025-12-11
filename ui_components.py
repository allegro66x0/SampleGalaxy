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
        self.hideAxis('bottom')
        self.setMouseEnabled(x=True, y=True)
        self.setMenuEnabled(False)
        
        # Experimental: GPU Acceleration
        try:
            self.useOpenGL(True)
            print("OpenGL acceleration enabled")
        except Exception as e:
            print(f"Failed to enable OpenGL: {e}")
        
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
            pxMode=True,
            hoverable=True,
            hoverPen=pg.mkPen('w'),
            hoverBrush=pg.mkBrush(255, 255, 255, 255),
            hoverSize=15
        )
        self.scatter.setZValue(10)
        self.addItem(self.scatter)
        
        # --- Data & State (Vectorized) ---
        self.ids_arr = np.array([])       # (N,) Path strings
        self.pos_arr = np.array([])       # (N, 2) Coordinates
        self.duration_arr = np.array([])  # (N,) Durations
        self.category_arr = np.array([])  # (N,) Categories (Strings)
        self.is_fav_arr = np.array([])    # (N,) Boolean
        self.base_brush_arr = np.array([])# (N, 4) Pre-calculated RGBA
        self.path_to_index = {}           # path -> index
        
        # Start Time Cache map (path -> start_time)
        self.start_times = {}

        self.points_data = [] # Keep raw data if needed, but avoiding usage in render loop
        self.playback_history = []
        self.current_pos = None
        
        self.filter_oneshot = False
        self.filter_favorites_only = False
        
        self.favorites = set()
        self.favorites_file = "favorites.json"
        
        # Visible State Cache
        self.visible_indices = None # Indices of currently visible points relative to full data
        
        self.category_colors = {
            "KICK": (255, 0, 50),       # Vivi Crimson
            "SNARE": (255, 100, 0),     # Vivid Orange
            "CLAP": (255, 50, 0),       # Vivid OrangeRed
            "HIHAT": (0, 255, 255),     # Cyan (Max Sat)
            "CRASH": (0, 150, 255),     # Vivid Blue
            "TOM": (160, 80, 20),       # Brownish
            "BASS": (150, 30, 255),     # Violet
            "GUITAR": (255, 230, 0),    # Vivid Gold
            "PIANO": (0, 255, 100),     # Vivid Green
            "LOOP": (0, 180, 0),        # Green
            "FX": (0, 120, 255),        # Blue
            "UNKNOWN": (100, 100, 100)  # Gray
        }
        self.visible_categories = set(self.category_colors.keys())
        self.visible_categories.add("TOP_LOOP")
        self.visible_categories.add("PERC")
        self.visible_categories.add("OTHER")

        # Events
        self.scene().sigMouseClicked.connect(self.on_scene_clicked)
        self.getPlotItem().hideAxis('bottom')
        self.getPlotItem().hideAxis('left')
        self.getPlotItem().getViewBox().setAspectLocked(True)
        
        self.drag_start_pos = None
        self.last_played_path = None
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.load_favorites() # Load favors initially

    # --- Favorites ---
    def load_favorites(self):
        data = utils.load_json(self.favorites_file)
        if data:
            self.favorites = {utils.normalize_path(p) for p in data}
        else:
            self.favorites = set()
        
        # Update boolean array if data is already loaded
        self._sync_fav_array()

    def _sync_fav_array(self):
        # Reset and Apply
        if len(self.ids_arr) > 0:
            self.is_fav_arr[:] = False
            # Find indices of favorites
            # This loop is fast enough for <1000 favorites usually
            # But relying on path_to_index is O(K) where K is num favorites
            for fav in self.favorites:
                idx = self.path_to_index.get(fav)
                if idx is not None:
                    self.is_fav_arr[idx] = True

    def save_favorites(self):
        utils.save_json(self.favorites_file, list(self.favorites))

    def add_favorite(self, path):
        path = utils.normalize_path(path)
        if path not in self.favorites:
            self.favorites.add(path)
            # instant update of array
            idx = self.path_to_index.get(path)
            if idx is not None:
                self.is_fav_arr[idx] = True
            
            print(f"Added to favorites: {path}")
            self.save_favorites()
            # If filtering by favorites, we need full update. 
            # If just highlighting, we can optimize but full update is fast now.
            self.update_plot()
            self.favorites_changed.emit()

    def remove_favorite(self, path):
        path = utils.normalize_path(path)
        if path in self.favorites:
            self.favorites.remove(path)
            # instant update of array
            idx = self.path_to_index.get(path)
            if idx is not None:
                self.is_fav_arr[idx] = False
                
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
        if self.filter_favorites_only != enabled:
            self.filter_favorites_only = enabled
            self.update_plot()
        
    def set_filter_oneshot(self, enabled):
        if self.filter_oneshot != enabled:
            self.filter_oneshot = enabled
            self.update_plot()
        
    def set_category_visibility(self, category, visible):
        if visible:
            if category not in self.visible_categories:
                self.visible_categories.add(category)
                self.update_plot()
        else:
            if category in self.visible_categories:
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
            
        # 1. Pre-process list to arrays
        # Use lists for building then convert to numpy
        ids = []
        pos = []
        durations = []
        categories = []
        colors = []
        
        self.path_to_index = {}
        self.start_times = {}
        
        cluster_palette = [(100, 100, 100), (150, 150, 150), (200, 200, 200)]
        # Map categories to colors
        cat_color_map = self.category_colors.copy()
        cat_color_map.update({"TOP_LOOP": (0, 180, 0), "PERC": (255, 50, 150), "OTHER": (180, 180, 180)})
        
        # Pre-calc centroid range
        centroids = [d.get('centroid', 0) for d in data if 'centroid' in d]
        if centroids:
             min_c, max_c = min(centroids), max(centroids)
        else:
             min_c, max_c = 0, 1
        if max_c == min_c: max_c += 1
        
        for i, item in enumerate(data):
            # Path
            path = item.get('path', '')
            path = utils.normalize_path(path) # Ensure normalization
            ids.append(path)
            self.path_to_index[path] = i
            self.start_times[path] = item.get('start_time', 0)
            
            # Pos
            pos.append([item['x'], item['y']])
            
            # Meta
            durations.append(item.get('duration', 0.0))
            
            cat = item.get('category', 'UNKNOWN')
            categories.append(cat)
            
            # Color Pre-calc
            # Logic: If specialized loop color, use that. Else if in map, use map. Else use cluster.
            c = (150, 150, 150)
            if cat == "LOOP" and 'centroid' in item:
                val = item['centroid']
                norm = (val - min_c) / (max_c - min_c)
                norm = max(0.0, min(1.0, norm))
                # Green to Yellow-Green (More vivid)
                r = int(0 + norm * 255)
                g = int(100 + norm * 155)
                b = int(0 + norm * 50)
                c = (r, g, b)
            elif cat in cat_color_map:
                c = cat_color_map[cat]
            else:
                cluster_id = item.get('cluster', 0)
                c = cluster_palette[cluster_id % len(cluster_palette)]
            
            colors.append(c + (255,)) # Add Alpha 255
            
        # Convert to numpy
        self.ids_arr = np.array(ids)
        self.pos_arr = np.array(pos, dtype=np.float32)
        self.duration_arr = np.array(durations, dtype=np.float32)
        self.category_arr = np.array(categories)
        self.base_brush_arr = np.array(colors, dtype=np.uint8)
        self.is_fav_arr = np.zeros(len(data), dtype=bool)
        
        self.points_data = data # keep reference just in case
        
        # Sync Favorites
        self._sync_fav_array()
        
        self.update_plot()
        
    def update_plot(self):
        if len(self.ids_arr) == 0:
            self.scatter.clear()
            return

        # 1. Construct Mask (Vectorized Filtering)
        mask = np.ones(len(self.ids_arr), dtype=bool)
        
        if self.filter_oneshot:
            mask &= (self.duration_arr <= 2.0)
            
        if self.filter_favorites_only:
            mask &= self.is_fav_arr
            
        # Category Filter (Vectorized using numpy.isin)
        # Checking against visible_set
        # It's faster to check what is NOT valid if visible set is large? 
        # Actually np.isin is optimized.
        # But category_arr is strings. Checking against a set/list of strings.
        # If all visible, skip.
        # If none visible, clear.
        if len(self.visible_categories) == 0:
            mask[:] = False
        else:
            # Only apply if not all categories are visible + extras (simple heuristic)
            # Actually just apply always unless full set
            # For correctness, use np.isin
            mask &= np.isin(self.category_arr, list(self.visible_categories))

        # 2. Apply Mask
        # self.visible_indices stores indices into the full arrays
        self.visible_indices = np.where(mask)[0]
        
        if len(self.visible_indices) == 0:
            self.scatter.setData(pos=[])
            return

        # 3. Prepare Visuals
        # Slice arrays
        vis_colors = self.base_brush_arr[self.visible_indices].copy()
        vis_favs = self.is_fav_arr[self.visible_indices]
        
        # Override favorite colors (Gold)
        # Assuming Gold is (255, 215, 0, 255)
        # We can do this efficiently
        if np.any(vis_favs):
            vis_colors[vis_favs] = [255, 215, 0, 255]
            
        # Sizes
        # Default 4, Fav 7, Hover 15 (handled by ScatterPlotItem itself usually, but we set base size)
        vis_sizes = np.full(len(self.visible_indices), 4, dtype=np.int8)
        if np.any(vis_favs):
            vis_sizes[vis_favs] = 7
            
        vis_pos = self.pos_arr[self.visible_indices]
        
        # 4. Draw
        # setData with arrays is much faster than addPoints with list of dicts
        self.scatter.setData(
            pos=vis_pos,
            size=vis_sizes,
            brush=vis_colors,
            pen=None,  # No border
            symbol='o' # vectorizing symbols supported? Yes, but usually single symbol is faster. Favs are 'star' in old code. 
            # Supporting different symbols in one setData requires 'symbol' array.
            # 'o' vs 'star'.
        )
        
        # If we really want stars for favorites, we need a list of symbols or symbol array.
        # ScatterPlotItem setData supports 'symbol' as array.
        # Let's try to keep it simple first 'o' for all, or construct symbol list.
        # Creating a list of strings is slow python loop?
        # np.full(..., 'o')
        # symbols = np.full(len(vis_sizes), 'o', dtype='<U10') # Unicode string array
        # symbols[vis_favs] = 'star'
        # self.scatter.setData(..., symbol=symbols)
        # Let's add that for completeness.
        
        symbols = np.full(len(self.visible_indices), 'o', dtype=object)
        if np.any(vis_favs):
            symbols[vis_favs] = 'star'
            
        self.scatter.setSymbol(symbols) # Use setSymbol or pass in setData? setData accepts symbol argument.
        # Note: setData(symbol=...) might expect list or array.
        
        # Optimization: Symbols might be slow to update if changed frequent.
        # But filtering is the heavy part.
        
    # --- Interaction ---
    def find_nearest_point(self, scene_pos, threshold=20):
        if self.visible_indices is None or len(self.visible_indices) == 0:
            return None

        # Map scene to view
        mouse_point = self.plotItem.vb.mapSceneToView(scene_pos)
        mouse_arr = np.array([mouse_point.x(), mouse_point.y()])
        
        # Get visible positions (sliced view)
        vis_pos = self.pos_arr[self.visible_indices]
        
        # Distances
        dists = np.linalg.norm(vis_pos - mouse_arr, axis=1)
        min_local_idx = np.argmin(dists)
        
        # Check threshold
        min_dist = dists[min_local_idx]
        
        # Convert pixel threshold roughly? Or use mapToScene logic again.
        # Using mapToScene for the nearest point is more accurate for screen distance.
        nearest_pos_view = QPointF(float(vis_pos[min_local_idx][0]), float(vis_pos[min_local_idx][1]))
        nearest_pos_scene = self.plotItem.vb.mapViewToScene(nearest_pos_view)
        pixel_dist = (nearest_pos_scene - scene_pos).manhattanLength()
        
        if pixel_dist < threshold:
            # Map back to global index
            global_idx = self.visible_indices[min_local_idx]
            path = self.ids_arr[global_idx]
            return path, nearest_pos_view, pixel_dist
            
        return None

    def on_scene_clicked(self, event):
        if event.button() != Qt.MouseButton.LeftButton: return
        result = self.find_nearest_point(event.scenePos())
        if result:
            path, data_pos, _ = result
            self.select_point(path)

    def select_point(self, path):
        path = utils.normalize_path(path)
        # O(1) Lookup
        idx = self.path_to_index.get(path)
        if idx is not None:
            self.last_played_path = path
            x, y = self.pos_arr[idx]
            self.update_history(QPointF(float(x), float(y)))
            
            start_time = self.start_times.get(path, 0)
            self.audio_engine.play(path, start_time)
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
        if self.current_pos is None or len(self.pos_arr) == 0: return
        curr = np.array(self.current_pos)
        
        # Calculate distance to ALL points (or visible points? usually random neighbor in global set is better)
        # Original logic used coords_array which was all points.
        dists = np.linalg.norm(self.pos_arr - curr, axis=1)
        nearest_indices = np.argsort(dists)[1:51]
        
        if len(nearest_indices) > 0:
            target_idx = np.random.choice(nearest_indices)
            path = self.ids_arr[target_idx]
            x, y = self.pos_arr[target_idx]
            
            self.update_history(QPointF(float(x), float(y)))
            start_time = self.start_times.get(path, 0)
            self.audio_engine.play(path, start_time)

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
                    # For Polyphonic/Scrubbing, we need start time.
                    # find_nearest_point returns path, but we have cache.
                    start_time = self.start_times.get(path, 0)
                    self.audio_engine.play(path, start_time, polyphonic=True)
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
                        start_time = self.start_times.get(path, 0)
                        self.audio_engine.play(path, start_time, polyphonic=True)
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
