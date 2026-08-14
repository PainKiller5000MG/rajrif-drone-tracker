import tkinter as tk
import os
import sys

from tkintermapview import TkinterMapView
from PIL import Image, ImageTk

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class DroneTrackerMap:
    def __init__(self, root, mount_lat, mount_long):
        self.window = tk.Toplevel(root)
        self.window.title("15 RAJRIF & 222 FD WKSP")
        self.window.geometry("800x600")
        #self.window.attributes("-topmost", False)  # always visible

        self.map_widget = TkinterMapView(self.window, width=900, height=600)
        self.map_widget.pack(fill="both", expand=True)

        self.map_widget.set_tile_server(
            "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
        )

        self.mount_icon = ImageTk.PhotoImage(
            Image.open(resource_path("LMG_icon.png")).resize((32, 32))
        )

        self.drone_icon = ImageTk.PhotoImage(
            Image.open(resource_path("drone_icon.png")).resize((32, 32))
        )

        self.mount_lat = mount_lat
        self.mount_long = mount_long

        self.mount_marker = self.map_widget.set_marker(
            self.mount_lat,
            self.mount_long,
            text="Mount",
            icon=self.mount_icon
        )

        self.map_widget.set_position(self.mount_lat, self.mount_long)
        self.map_widget.set_zoom(12)

        self.drone_marker = None

    def update_drone(self, lat, lon):

        if lat is None or lon is None:
            if self.drone_marker:
                self.drone_marker.delete()
                self.drone_marker = None
            return

        if self.drone_marker is None:
            self.drone_marker = self.map_widget.set_marker(
                lat, lon,
                text="Drone",
                icon=self.drone_icon
            )
        else:
            self.drone_marker.set_position(lat, lon)

# -----------------------------
# Example usage
# -----------------------------
'''
if __name__ == "__main__":
    import tkinter as tk

    root = tk.Tk()

    app = DroneTrackerMap(root, 30.7333, 76.7794)

    gps_data = [
        (30.7350, 76.7800),
        (30.7365, 76.7815),
        (30.7375, 76.7825),
        (None, None),
        (30.7385, 76.7835),
        (30.7395, 76.7845),
    ]

    state = {"i": 0}   # ✅ FIX: mutable state container

    def update_loop():
        i = state["i"]

        lat, lon = gps_data[i]
        app.update_drone(lat, lon)

        state["i"] = (i + 1) % len(gps_data)

        root.after(5000, update_loop)

    update_loop()
    root.mainloop()
'''