import os, asyncio
import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor2D
from vtkmodules.vtkRenderingCore import vtkRenderer
from omegaconf import OmegaConf
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3
from pyvista.trame.ui import plotter_ui
from diffusion import step_diff, step_wave

# ------------ 設定読み込み ----------------
param = OmegaConf.load("param.yaml")
NX, NY = param.nx, param.ny

# ------------ 初期データ生成 ----------------
def make_initial_u():
    u0 = np.zeros((NX, NY), np.float32)
    u0[slice(*param.init_xlim), slice(*param.init_ylim)] = param.init_hotvalue
    return u0

u = make_initial_u()

def make_grid(u):
    x = np.arange(NX, dtype=np.float32)
    y = np.arange(NY, dtype=np.float32)
    x, y = np.meshgrid(x, y, indexing="ij")
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
    grid["u"] = z_flat

# ------------ カメラ初期化 ----------------
def init_camera(plotter, param):
    camera = plotter.camera
    camera.position = param.camera_position
    camera.focal_point = param.camera_focal_point
    camera.up = param.camera_view_up
    camera.zoom = param.get("camera_zoom", 1.0)
    plotter.camera = camera

# ------------ 境界線の直方体 ----------------
def add_bounding_box(plotter, nx, ny):
    box = pv.Box(bounds=(0, nx, 0, ny, 0, param.init_hotvalue))
    plotter.add_mesh(
        box.extract_all_edges(),
        color="black",
        line_width=2,
        style="wireframe",
        opacity=1.0
    )
    

# ------------ Plotter 初期化 ----------------
plotter = pv.Plotter(off_screen=True, border=True, window_size=(400, 400))
plotter.add_mesh(
    grid,
    scalars="u",
    cmap=param.cmap,
    clim=param.clim,
    scalar_bar_args={"title": "温度だお", "vertical": True}
)

add_bounding_box(plotter, NX, NY)
plotter.show_bounds(
    xlabel="X", ylabel="Y", zlabel="U",
    color="black", grid=False
)
init_camera(plotter, param)

# ------------ Trame ----------------
server = get_server()
server.client_type = "vue3" 
state, ctrl = server.state, server.controller
state.running = False

def step_loop():
    if state.running:
        global u
        for _ in range(param.step_per_frame):
            #step_diff/wave
            u = step_diff(u)
        update_grid(u)
        ctrl.view_update()
        asyncio.get_event_loop().call_later(0.1, step_loop)

@ctrl.add("play_pause")
def toggle_play():
    state.running = not state.running
    state.dirty("running")
    ctrl.view_update()
    if state.running:
        asyncio.get_event_loop().call_later(0.1, step_loop)

@ctrl.add("reset")
def reset():
    global u
    u = make_initial_u()
    update_grid(u)
    state.running = False
    state.dirty("running")
    ctrl.view_update()

# ------------ UI ----------------
with SinglePageLayout(server) as layout:
    layout.title.set_text("Diffusion Demo")
    layout.icon.clear()
    with layout.toolbar:
        v3.VBtn(icon=True, click=ctrl.play_pause, children=[
            v3.VIcon("mdi-pause" if state.running else "mdi-play")
        ])
        v3.VBtn(icon=True, click=ctrl.reset, children=[
            v3.VIcon("mdi-reload")
        ])
    with layout.content:
        view = plotter_ui(plotter, toolbar=False)
        ctrl.view_update = view.update

# ------------ 起動 ----------------
if __name__ == "__main__":
    is_render = "PORT" in os.environ
    server.start(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        open_browser=not is_render
    )
