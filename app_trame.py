import os, threading
import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from pyvista.trame.ui import plotter_ui
from diffusion import step

# ---- 初期データ ----
NX, NY = 100, 100
u = np.zeros((NY, NX), dtype=np.float32)
u[NY//2, NX//2] = 10.0  # 中央熱源

# ---- PyVista グリッド ----
grid = pv.ImageData(dimensions=(NX, NY, 1), spacing=(1, 1, 1), origin=(0, 0, 0))
grid.point_data["u"] = u.ravel(order="F")

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(grid, scalars="u", cmap="inferno")
plotter.view_xy()

# ---- Trame サーバ ----
server = get_server()
state, ctrl = server.state, server.controller

# 再生ループ用フラグ
state.running = False
def _loop():
    if state.running:
        # 1ステップ計算
        global u
        u = step(u)
        grid.point_data["u"] = u.ravel(order="F")
        ctrl.view_update()
        # 0.1 秒後に再度スケジュール
        threading.Timer(0.1, _loop).start()

@ctrl.add("play_pause")
def toggle_play():
    state.running = not state.running
    if state.running:
        _loop()

with SinglePageLayout(server) as layout:
    layout.title.set_text("Diffusion Demo")
    with layout.toolbar:
        layout.toolbar.add_btn(icon="mdi-play-pause", click=ctrl.play_pause)
    with layout.content:
        view = plotter_ui(plotter)
        ctrl.view_update = view.update

# Render 用ポート取得
server.start(address="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
