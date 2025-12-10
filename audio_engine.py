from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl, QTimer, QObject
import os

class AudioEngine(QObject):
    def __init__(self, pool_size=8):
        super().__init__()
        
        # Main Player (Single Voice, for normal preview)
        self.main_output = QAudioOutput()
        self.main_player = QMediaPlayer()
        self.main_player.setAudioOutput(self.main_output)
        self.main_output.setVolume(0.8)
        
        # Polyphonic Scrubbing Pool
        self.pool_size = pool_size
        self.scrub_pool = []
        self.scrub_outputs = [] # Keep references to prevent GC
        self.pool_index = 0
        
        for _ in range(self.pool_size):
            p = QMediaPlayer()
            o = QAudioOutput()
            p.setAudioOutput(o)
            o.setVolume(0.8)
            self.scrub_pool.append(p)
            self.scrub_outputs.append(o)

    def get_main_player(self):
        return self.main_player

    def play(self, file_path, start_time=0, polyphonic=False):
        abs_path = os.path.abspath(file_path)
        url = QUrl.fromLocalFile(abs_path)
        start_ms = int(start_time * 1000)

        if polyphonic:
            # Sound Pool (Round Robin)
            player = self.scrub_pool[self.pool_index]
            self.pool_index = (self.pool_index + 1) % self.pool_size
            
            # Crossfade logic: Stop other playing sounds after 0.3s
            for p in self.scrub_pool:
                if p is not player and p.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                    QTimer.singleShot(300, p.stop)

            if player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                 player.stop()
                 
            player.setSource(url)
            if start_ms > 0:
                player.setPosition(start_ms)
            player.play()
            
        else:
            # Main Player
            self.main_player.stop()
            self.main_player.setSource(url)
            
            if start_ms > 0:
                self.main_player.setPosition(start_ms)
                
            self.main_player.play()
            
        print(f"Playing: {file_path} (Start: {start_time:.2f}s, Poly: {polyphonic})")

    def stop_all_scrubbing(self):
        """Stop all polyphonic players immediately."""
        for p in self.scrub_pool:
            if p.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                p.stop()
