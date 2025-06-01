import os, asyncio
import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3
from pyvista.trame.ui import plotter_ui
from diffusion import step

NX, NY = 100, 100
def make_initial_u():
    u0 = np.zeros((NY, NX), np.float32)
    u0[40:60, 40:60] = 10.0
    return u0

u = make_initial_u()

# ------------ StructuredGrid --------------
def make_grid(u):
    x, y = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    z = u.copy()
    g = pv.StructuredGrid(x, y, z)
    g["u"] = u.ravel(order="F")
    return g

grid = make_grid(u)

def update_grid(u):
    z_flat = u.ravel(order="F")
    pts = grid.points.copy()
    pts[:, 2] = z_flat
    grid.points = pts
    grid["u"] = z_flat          # 色も更新

# ------------ Plotter ---------------------
plotter = pv.Plotter(off_screen=True)
actor = plotter.add_mesh(
    grid, scalars="u", cmap="viridis", clim=[0, 10],
    scalar_bar_args={"title": "温度", "vertical": True}
)
plotter.view_xy();  plotter.camera.zoom(1.5)

# ------------ Trame -----------------------
server = get_server()
state, ctrl = server.state, server.controller
state.running = False

def step_loop():
    if state.running:
        global u
        for _ in range(5):     
            u = step(u)
        update_grid(u)
        ctrl.view_update()
        asyncio.get_event_loop().call_later(0.1, step_loop)

@ctrl.add("play_pause")
def toggle_play():
    state.running = not state.running
    state.dirty("running");  ctrl.view_update()
    if state.running:
        asyncio.get_event_loop().call_later(0.1, step_loop)

@ctrl.add("reset")
def reset():
    global u
    u = make_initial_u()
    update_grid(u)
    state.running = False
    state.dirty("running");  ctrl.view_update()

# ------------ UI --------------------------
with SinglePageLayout(server) as layout:
    layout.title.set_text("Diffusion Demo")
    layout.icon.clear()                             # 右上の自動ボタン除去
    with layout.toolbar:
        v3.VBtn(icon=True, click=ctrl.play_pause, children=[
            v3.VIcon("mdi-pause" if state.running else "mdi-play")
        ])
        v3.VBtn(icon=True, click=ctrl.reset, children=[ v3.VIcon("mdi-reload") ])
    with layout.content:
        view = plotter_ui(plotter, toolbar=False)   # ← 内部ツールバー無効
        ctrl.view_update = view.update

server.start(address="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
