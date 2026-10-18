"""Zoomable image canvas with overlay blending and a pixel read-out."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def to_qimage(bgr: np.ndarray) -> QImage:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, w * 3, QImage.Format_BGR888).copy()


def to_pixmap(bgr: np.ndarray) -> QPixmap:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]                           # fromImage converts BGR888 into its own buffer: no extra copy
    return QPixmap.fromImage(QImage(bgr.data, w, h, w * 3, QImage.Format_BGR888))


ZOOM_STEP = 1.25                                   # one mouse-wheel notch, and one press of the zoom buttons
MIN_ZOOM, MAX_ZOOM = 0.05, 10.0                    # 5 % .. 1000 %


class ImageView(QGraphicsView):
    """Base image with an optional pixel-aligned overlay. Wheel zooms around the
    cursor, drag pans, double-click fits."""
    pixel_hovered = Signal(int, int)
    view_changed = Signal()
    line_drawn = Signal(float, float, float, float)   # draw mode: end points in image coordinates

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.base = QGraphicsPixmapItem()
        self.overlay = QGraphicsPixmapItem()
        for item in (self.base, self.overlay):
            item.setTransformationMode(Qt.FastTransformation)
            self.scene().addItem(item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setMouseTracking(True)
        self._fitted = True
        self._base_source: np.ndarray | None = None   # array the base pixmap was made from
        self._draw = False
        self._start: QPointF | None = None
        self._rubber: QGraphicsLineItem | None = None
        self._vectors: list = []
        self._syncing = False

    def set_images(self, base: np.ndarray | None, overlay: np.ndarray | None = None, opacity: float = 1.0,
                   keep_view: bool = False) -> None:
        old = self.base.pixmap().size()
        if base is not self._base_source:            # switching views of one image keeps the converted base
            self.base.setPixmap(to_pixmap(base) if base is not None else QPixmap())
            self._base_source = base
        self.overlay.setPixmap(to_pixmap(overlay) if overlay is not None else QPixmap())
        self.overlay.setOpacity(opacity)
        self.scene().setSceneRect(self.base.boundingRect())
        if not keep_view or old != self.base.pixmap().size():
            self.fit()

    # ------------------------------------------------------------------ examiner-drawn lines (geometry tool)
    def set_draw_mode(self, on: bool) -> None:
        """Left-drag draws a line instead of panning; hold Ctrl to pan."""
        self._draw = on
        self.setDragMode(QGraphicsView.NoDrag if on else QGraphicsView.ScrollHandDrag)
        self.viewport().setCursor(Qt.CrossCursor if on else Qt.OpenHandCursor)

    def set_lines(self, lines: list[tuple[float, float, float, float, tuple[int, int, int]]],
                  points: list[tuple[float, float]]) -> None:
        """Lines (x1, y1, x2, y2, bgr) extended across the scene, and the common points as crosses. The scene
        grows to show common points up to three image sizes outside the frame."""
        for item in self._vectors:
            self.scene().removeItem(item)
        self._vectors = []
        frame = self.base.boundingRect()
        reach = 10 * max(frame.width(), frame.height(), 1)
        for x1, y1, x2, y2, (b, g, r) in lines:
            line = QLineF(x1, y1, x2, y2)
            if line.length() < 1:
                continue
            unit = line.unitVector()
            dx, dy = unit.dx() * reach, unit.dy() * reach
            pen = QPen(QColor(r, g, b), 2)
            pen.setCosmetic(True)                     # constant width on screen at every zoom
            self._vectors.append(self.scene().addLine(x1 - dx, y1 - dy, x1 + dx, y1 + dy, pen))
            for x, y in ((x1, y1), (x2, y2)):
                self._vectors.append(self.scene().addEllipse(x - 3, y - 3, 6, 6, pen))
        rect = QRectF(frame)
        pen = QPen(QColor(255, 255, 255), 2)
        pen.setCosmetic(True)
        for x, y in points:
            x = min(max(x, -3 * frame.width()), 4 * frame.width())
            y = min(max(y, -3 * frame.height()), 4 * frame.height())
            size = max(frame.width(), frame.height()) / 80
            self._vectors.append(self.scene().addLine(x - size, y - size, x + size, y + size, pen))
            self._vectors.append(self.scene().addLine(x - size, y + size, x + size, y - size, pen))
            rect = rect.united(QRectF(x - 2 * size, y - 2 * size, 4 * size, 4 * size))
        self.scene().setSceneRect(rect)

    def mousePressEvent(self, event):
        if self._draw and event.button() == Qt.LeftButton and not event.modifiers() & Qt.ControlModifier:
            self._start = self.mapToScene(event.position().toPoint())
            pen = QPen(QColor(255, 255, 255), 1, Qt.DashLine)
            pen.setCosmetic(True)
            self._rubber = self.scene().addLine(QLineF(self._start, self._start), pen)
            return
        if self._draw:                               # Ctrl + drag pans while drawing is on
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._rubber is not None:
            end = self.mapToScene(event.position().toPoint())
            self.scene().removeItem(self._rubber)
            self._rubber = None
            if QLineF(self._start, end).length() * self.zoom_factor() > 8:      # ignore clicks
                self.line_drawn.emit(self._start.x(), self._start.y(), end.x(), end.y())
            return
        super().mouseReleaseEvent(event)
        if self._draw:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)

    def shows(self, image: np.ndarray | None) -> bool:
        """True while `image` itself is the base layer (not a plot or another picture)."""
        return image is not None and image is self._base_source

    def set_opacity(self, opacity: float) -> None:
        self.overlay.setOpacity(opacity)

    def fit(self) -> None:
        if not self.base.pixmap().isNull():
            self.fitInView(self.sceneRect() if self._vectors else self.base.boundingRect(), Qt.KeepAspectRatio)
            self._fitted = True
            self.view_changed.emit()

    def actual_size(self) -> None:
        self.resetTransform()
        self._fitted = False
        self.view_changed.emit()

    def zoom_factor(self) -> float:
        return self.transform().m11()

    def link(self, peer: "ImageView") -> None:
        """Keep zoom and position identical in two views."""
        for a, b in ((self, peer), (peer, self)):
            a.view_changed.connect(lambda a=a, b=b: a._push(b))
            a.horizontalScrollBar().valueChanged.connect(lambda _, a=a, b=b: a._push(b))
            a.verticalScrollBar().valueChanged.connect(lambda _, a=a, b=b: a._push(b))

    def _push(self, other: "ImageView") -> None:
        if self._syncing or other._syncing or not other.isVisible():
            return
        other._syncing = True
        other.setTransform(self.transform())
        other.centerOn(self.mapToScene(self.viewport().rect().center()))
        other._fitted = self._fitted               # a fitted pair must keep re-fitting when the window resizes
        other._syncing = False

    def zoom_step(self, steps: int, at_cursor: bool = True) -> None:
        """Zoom in (steps > 0) or out by whole wheel notches, within 5 % .. 1000 %. A fitted view that is already
        outside that range (a huge or a tiny image) is never pushed further out, only back towards it."""
        current = self.zoom_factor()
        lo, hi = min(MIN_ZOOM, current), max(MAX_ZOOM, current)
        target = min(max(current * ZOOM_STEP ** steps, lo), hi)
        if target == current:
            return
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse if at_cursor else QGraphicsView.AnchorViewCenter)
        self.scale(target / current, target / current)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._fitted = False
        self.view_changed.emit()

    def wheelEvent(self, event):
        if event.angleDelta().y() == 0:              # horizontal wheel / touchpad swipe
            return
        self.zoom_step(1 if event.angleDelta().y() > 0 else -1)

    def mouseDoubleClickEvent(self, event):
        if not self._draw:
            self.fit() if not self._fitted else self.actual_size()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        p: QPointF = self.mapToScene(event.position().toPoint())
        if self._rubber is not None:
            self._rubber.setLine(QLineF(self._start, p))
        self.pixel_hovered.emit(int(p.x()), int(p.y()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fitted:
            self.fit()
