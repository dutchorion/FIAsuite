"""Zoomable image canvas with overlay blending and a pixel read-out."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def to_qimage(bgr: np.ndarray) -> QImage:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, w * 3, QImage.Format_BGR888).copy()


def to_pixmap(bgr: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(to_qimage(bgr))


class ImageView(QGraphicsView):
    """Base image with an optional pixel-aligned overlay. Wheel zooms around the
    cursor, drag pans, double-click fits."""
    pixel_hovered = Signal(int, int)
    view_changed = Signal()

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
        self._peer: ImageView | None = None
        self._syncing = False

    def set_images(self, base: np.ndarray | None, overlay: np.ndarray | None = None, opacity: float = 1.0,
                   keep_view: bool = False) -> None:
        old = self.base.pixmap().size()
        self.base.setPixmap(to_pixmap(base) if base is not None else QPixmap())
        self.overlay.setPixmap(to_pixmap(overlay) if overlay is not None else QPixmap())
        self.overlay.setOpacity(opacity)
        self.scene().setSceneRect(self.base.boundingRect())
        if not keep_view or old != self.base.pixmap().size():
            self.fit()

    def set_opacity(self, opacity: float) -> None:
        self.overlay.setOpacity(opacity)

    def fit(self) -> None:
        if not self.base.pixmap().isNull():
            self.fitInView(self.base, Qt.KeepAspectRatio)
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
        self._peer, peer._peer = peer, self
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
        other._fitted = False
        other._syncing = False

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        if 0.02 < self.zoom_factor() * factor < 64:
            self.scale(factor, factor)
            self._fitted = False
            self.view_changed.emit()

    def mouseDoubleClickEvent(self, event):
        self.fit() if not self._fitted else self.actual_size()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        p: QPointF = self.mapToScene(event.position().toPoint())
        self.pixel_hovered.emit(int(p.x()), int(p.y()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fitted:
            self.fit()
