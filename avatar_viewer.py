"""Panda3D desktop renderer for a procedural, stylized 3D head.

It reads the JSON created by avatar_generator.py. There is no GLB asset
requirement, but this is an illustrative 3D character, not a realistic scan.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (AmbientLight, DirectionalLight, Geom, GeomNode,
    GeomTriangles, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
    LineSegs, NodePath, Vec4, WindowProperties)


def _rgba(rgb, alpha=1):
    return Vec4(*(max(0, min(255, float(c))) / 255 for c in rgb), alpha)


def _sphere(parent, name, scale, pos, color, lat_steps=22, lon_steps=32,
            theta_max=math.pi):
    """Generate actual 3D triangle geometry, including optional upper cap."""
    data = GeomVertexData(name, GeomVertexFormat.getV3n3(), Geom.UHStatic)
    vertex = GeomVertexWriter(data, 'vertex')
    normal = GeomVertexWriter(data, 'normal')
    sx, sy, sz = scale
    for j in range(lat_steps + 1):
        theta = theta_max * j / lat_steps
        for i in range(lon_steps + 1):
            phi = 2 * math.pi * i / lon_steps
            u, v, z = math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)
            vertex.addData3(u*sx, v*sy, z*sz)
            n_x, n_y, n_z = u / sx, v / sy, z / sz
            norm = math.sqrt(n_x*n_x + n_y*n_y + n_z*n_z)
            normal.addData3(n_x/norm, n_y/norm, n_z/norm)
    tris = GeomTriangles(Geom.UHStatic)
    for j in range(lat_steps):
        for i in range(lon_steps):
            a = j * (lon_steps + 1) + i
            b = a + lon_steps + 1
            # Winding outwards for standard spherical coordinates.
            tris.addVertices(a, b, a+1)
            tris.addVertices(a+1, b, b+1)
    geom = Geom(data)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    np = parent.attachNewNode(node)
    np.setPos(*pos)
    np.setColor(_rgba(color))
    np.setTwoSided(True)
    return np


def _curve(parent, name, points, color, thickness=5):
    lines = LineSegs(name)
    lines.setThickness(thickness)
    lines.setColor(_rgba(color))
    for idx, p in enumerate(points):
        if idx == 0:
            lines.moveTo(*p)
        else:
            lines.drawTo(*p)
    return parent.attachNewNode(lines.create())


class AvatarViewer:
    """Panda3D viewer supporting the API used by demo.py and main.py.

    update_expression() changes facial controls; update() processes one
    application event-loop step. Call update() repeatedly from your loop.
    """

    def __init__(self, avatar_path: str | Path):
        params = json.loads(Path(avatar_path).read_text(encoding="utf-8"))
        self.base = ShowBase(windowType="onscreen")
        self.base.disableMouse()
        self.base.setBackgroundColor(0.13, 0.16, 0.23, 1)
        self.closed = False
        self.base.accept("escape", self.close)
        self.base.accept("window-event", self._window_event)
        self.base.win.setClearColor(Vec4(.13, .16, .23, 1))
        props = WindowProperties()
        props.setTitle("Personalized 3D Emoji Avatar")
        props.setSize(800, 720)
        self.base.win.requestProperties(props)

        self.base.camera.setPos(0, -7.2, 0.2)
        self.base.camera.lookAt(0, 0, 0)
        amb = AmbientLight("ambient")
        amb.setColor((.55, .55, .6, 1))
        self.base.render.setLight(self.base.render.attachNewNode(amb))
        sun = DirectionalLight("key")
        sun.setColor((.85, .8, .75, 1))
        light_np = self.base.render.attachNewNode(sun)
        light_np.setHpr(-25, -35, 0)
        self.base.render.setLight(light_np)

        self.root = self.base.render.attachNewNode('head')
        self.skin = params.get("skin_rgb", [196, 148, 118])
        self.hair = params.get("hair_rgb", [52, 37, 31])
        self.iris = params.get("eye_rgb", [58, 77, 92])
        self.face_width = max(.7, min(1.3, float(params.get("face_width", 1))))
        self.face_height = max(.8, min(1.2, float(params.get("face_height", 1))))
        self.eye_spacing = max(.7, min(1.4, float(params.get("eye_spacing", 1))))
        self.nose_size = max(.7, min(1.4, float(params.get("nose_size", 1))))
        self.mouth_width = max(.65, min(1.3, float(params.get("mouth_width", 1))))
        self._make_head()
        self.controls = {"smile":0, "frown":0, "mouth_open":0, "brow_raise":0,
                         "brow_down":0, "blink_left":0, "blink_right":0}
        self._mouth_shape = None
        self.update_expression({})

    def _make_head(self):
        w, h = self.face_width, self.face_height
        _sphere(self.root, "headmesh", (1.0*w, .68, 1.27*h), (0,0,0), self.skin)
        for side in [-1, 1]:
            _sphere(self.root, "ear", (.19, .13, .29), (side*.99*w, -.012, -.19), self.skin)
        # Top hemisphere (hair) is an actual 3D surface, not a flat image.
        _sphere(self.root, "haircap", (1.025*w, .705, 1.29*h), (0,.018,.045),
                self.hair, theta_max=1.20)
        _sphere(self.root, "nose", (.15*self.nose_size, .20, .24*self.nose_size),
                (0,-.69,.005), self.skin)
        self.eyes = []
        self.brows = []
        for side in [-1, 1]:
            x = side*.39*self.eye_spacing*w
            eyeball = _sphere(self.root, "eyewhite", (.235, .105, .18),
                              (x,-.595,.30*h), [252,248,244])
            _sphere(eyeball, "iris", (.08, .035, .095), (0,-.87,0), self.iris)
            _sphere(eyeball, "pupil", (.045,.019,.056), (0,-1.18,0), [17,18,24])
            self.eyes.append(eyeball)
            brow = self.root.attachNewNode("brow")
            _curve(brow,"browline",[(x-.20,-.72,.59*h), (x,-.78,.64*h),
                                     (x+.20,-.72,.59*h)], self.hair, 7)
            self.brows.append(brow)
        self.mouth = self.root.attachNewNode("mouth")

    def _draw_mouth(self, smile, frown, mouth_open):
        if self.mouth is not None:
            self.mouth.removeNode()
        self.mouth = self.root.attachNewNode("mouth")
        y = -.645
        width = .39*self.mouth_width
        curvature = .25*smile - .18*frown
        points = [(width*(i/10), y-.04*(1-abs(i/10)), -.46 - curvature*(1-abs(i/10)**2))
                  for i in range(-10,11)]
        _curve(self.mouth,"lip",points,[94,40,54],8)
        if mouth_open > .07:
            _sphere(self.mouth,"openmouth",(.26*self.mouth_width,
                .035, .18*mouth_open), (0,y-.025,-.54), [89,33,48])

    def update_expression(self, values: dict[str, float]):
        for key, val in values.items():
            if key in self.controls:
                self.controls[key] = max(0., min(1., float(val)))
        c = self.controls
        for eye, blink in zip(self.eyes, (c['blink_left'], c['blink_right'])):
            eye.setSz(max(.07, 1.0-blink*.93))
        for brow in self.brows:
            brow.setZ(.15*c['brow_raise']-.11*c['brow_down'])
        shape = tuple(round(c[k],2) for k in ('smile','frown','mouth_open'))
        if shape != self._mouth_shape:
            self._draw_mouth(c['smile'],c['frown'],c['mouth_open'])
            self._mouth_shape = shape

    def set_head_rotation(self, heading=0, pitch=0, roll=0):
        self.root.setHpr(float(heading), float(pitch), float(roll))

    def update(self):
        if not self.closed:
            self.base.taskMgr.step()

    def should_close(self):
        return self.closed or self.base.win is None

    def close(self):
        self.closed = True

    def _window_event(self, window):
        if not window or not window.getProperties().getOpen():
            self.closed = True

    def destroy(self):
        self.base.destroy()
