import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider, 
                           QHBoxLayout, QGroupBox, QPushButton, QCheckBox, QScrollArea,
                           QAbstractItemView, QLabel, QListWidgetItem)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

# Modular imports
from ui_components import GalaxyPlotWidget, FavoritesListWidget
import utils

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sample Galaxy")
        self.resize(1200, 800)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Sidebar ---
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. Display Options
        disp_group = QGroupBox("Display Options")
        disp_layout = QVBoxLayout()
        self.one_shot_checkbox = QCheckBox("One-Shot (< 2s)")
        self.one_shot_checkbox.stateChanged.connect(self.on_oneshot_changed)
        disp_layout.addWidget(self.one_shot_checkbox)
        
        self.fav_checkbox = QCheckBox("Show Favorites Only (AND)")
        self.fav_checkbox.stateChanged.connect(self.on_fav_filter_changed)
        disp_layout.addWidget(self.fav_checkbox)
        disp_group.setLayout(disp_layout)
        self.sidebar_layout.addWidget(disp_group)
        
        # 2. Favorites List
        fav_group = QGroupBox("Favorites List")
        fav_layout = QVBoxLayout()
        self.fav_list_widget = FavoritesListWidget()
        self.fav_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.fav_list_widget.itemClicked.connect(self.on_fav_item_clicked)
        self.fav_list_widget.files_dropped.connect(self.on_files_dropped)
        fav_layout.addWidget(self.fav_list_widget)
        fav_group.setLayout(fav_layout)
        self.sidebar_layout.addWidget(fav_group)
        
        # 3. Categories
        cat_group = QGroupBox("Tags (Categories)")
        cat_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_all.clicked.connect(self.on_select_all)
        btn_none.clicked.connect(self.on_deselect_all)
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_none)
        cat_layout.addLayout(btn_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        
        # Create Plot Widget first to get categories
        self.plot_widget = GalaxyPlotWidget()
        self.category_checkboxes = {}
        
        for cat, color in self.plot_widget.category_colors.items():
            color_hex = QColor(*color).name()
            cb = QCheckBox(cat)
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {color_hex}; font-weight: bold; }}")
            cb.stateChanged.connect(lambda state, c=cat: self.on_category_changed(c, state))
            self.scroll_layout.addWidget(cb)
            self.category_checkboxes[cat] = cb
            
        self.scroll_layout.addStretch()
        self.scroll_content.setLayout(self.scroll_layout)
        scroll.setWidget(self.scroll_content)
        cat_layout.addWidget(scroll)
        cat_group.setLayout(cat_layout)
        self.sidebar_layout.addWidget(cat_group)
        self.main_layout.addWidget(self.sidebar)
        
        # --- Right Side (Plot + Seek) ---
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.addWidget(self.plot_widget)
        
        self.seek_layout = QHBoxLayout()
        self.seek_layout.setContentsMargins(10, 0, 10, 10)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_layout.addWidget(self.seek_slider)
        self.right_layout.addLayout(self.seek_layout)
        self.main_layout.addWidget(self.right_widget)
        
        # Connections
        self.player = self.plot_widget.get_player()
        self.player.positionChanged.connect(self.update_slider)
        self.player.durationChanged.connect(self.update_duration)
        self.seek_slider.sliderMoved.connect(self.set_position)
        self.seek_slider.sliderPressed.connect(self.slider_pressed)
        self.seek_slider.sliderReleased.connect(self.slider_released)
        self.is_slider_pressed = False
        
        self.plot_widget.favorites_changed.connect(self.update_favorites_list)
        
        # Init Load
        self.plot_widget.load_data("database.json")
        self.update_favorites_list()

    # --- Favorites Logic ---
    def on_files_dropped(self, files):
        for f in files:
            self.plot_widget.add_favorite(f)
            
    def on_fav_item_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        self.plot_widget.select_point(path)
        
    def update_favorites_list(self):
        self.fav_list_widget.clear()
        for path in sorted(self.plot_widget.favorites):
            item = QListWidgetItem(self.fav_list_widget)
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(5, 2, 5, 2)
            
            name = QLabel(utils.normalize_path(path).split(os.sep)[-1])
            layout.addWidget(name)
            layout.addStretch()
            
            del_btn = QPushButton("×")
            del_btn.setFixedSize(20, 20)
            del_btn.setStyleSheet("QPushButton { border: none; font-weight: bold; color: #888; } QPushButton:hover { color: red; }")
            del_btn.clicked.connect(lambda _, p=path: self.plot_widget.remove_favorite(p))
            layout.addWidget(del_btn)
            
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, path)
            
            self.fav_list_widget.addItem(item)
            self.fav_list_widget.setItemWidget(item, widget)

    # --- Filter Logic ---
    def on_oneshot_changed(self, state):
        self.plot_widget.set_filter_oneshot(state == 2)
        
    def on_fav_filter_changed(self, state):
        self.plot_widget.set_filter_favorites(state == 2)
        
    def on_category_changed(self, category, state):
        self.plot_widget.set_category_visibility(category, state == 2)
        
    def on_select_all(self):
        for cb in self.category_checkboxes.values(): cb.setChecked(True)
        self.plot_widget.set_all_categories_visibility(True)
        
    def on_deselect_all(self):
        for cb in self.category_checkboxes.values(): cb.setChecked(False)
        self.plot_widget.set_all_categories_visibility(False)

    # --- Seeker Logic ---
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
        self.player.play()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
