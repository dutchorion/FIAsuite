"""Dark theme shared by all windows."""
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

BG, PANEL, RAISED, BORDER = "#111113", "#18191b", "#212225", "#2e3135"
TEXT, MUTED, ACCENT = "#edeef0", "#9b9ea6", "#3e63dd"
SEVERITY_COLORS = {"info": "#8b8d98", "note": "#f5d90a", "warn": "#f76b15", "alert": "#e5484d"}

STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI'; font-size: 10pt; }}
QMainWindow::separator {{ background: {BORDER}; width: 5px; height: 5px; }}
QMainWindow::separator:hover {{ background: {ACCENT}; }}
QToolBar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; padding: 4px; spacing: 4px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 6px 12px; }}
QToolButton:hover {{ background: {RAISED}; border-color: {BORDER}; }}
QToolButton:disabled {{ color: #5a5d66; }}
QToolButton#primary {{ background: {ACCENT}; color: white; font-weight: 600; }}
QToolButton#primary:disabled {{ background: #2a3350; color: #7c829a; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {PANEL}; padding: 6px 10px; border-bottom: 1px solid {BORDER}; }}
QTreeWidget, QTableWidget, QTextBrowser, QListWidget {{ background: {PANEL}; border: none; outline: none; }}
QTreeWidget::item {{ padding: 4px 2px; }}
QTreeWidget::item:selected, QTableWidget::item:selected {{ background: {ACCENT}; color: white; }}
QHeaderView::section {{ background: {RAISED}; border: none; border-right: 1px solid {BORDER}; padding: 5px 8px; }}
QTableWidget {{ gridline-color: {BORDER}; }}
QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:default {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 5px; padding: 4px 8px; }}
QComboBox QAbstractItemView {{ background: {RAISED}; selection-background-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; border-radius: 7px; background: {ACCENT}; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }} QScrollBar:horizontal {{ background: {BG}; height: 10px; }}
QScrollBar::handle {{ background: {BORDER}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QProgressBar {{ background: {RAISED}; border: none; border-radius: 3px; height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#card:hover {{ border-color: {ACCENT}; }}
QLabel#muted {{ color: {MUTED}; }} QLabel#h1 {{ font-size: 16pt; font-weight: 600; }} QLabel#h2 {{ font-size: 11pt; font-weight: 600; }}
QLabel {{ background: transparent; }}
QCheckBox {{ background: transparent; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid #5a5d66; border-radius: 4px; background: {RAISED}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: none; }}
QCheckBox::indicator:disabled {{ border-color: {BORDER}; }}
QListWidget::item {{ padding: 4px 2px; }}
QListWidget::indicator {{ width: 15px; height: 15px; border: 1px solid #5a5d66; border-radius: 4px; background: {RAISED}; }}
QListWidget::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSplitter::handle {{ background: {BORDER}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER}; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, BG), (QPalette.Base, PANEL), (QPalette.AlternateBase, RAISED),
                        (QPalette.Text, TEXT), (QPalette.WindowText, TEXT), (QPalette.Button, RAISED),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT), (QPalette.ToolTipBase, RAISED),
                        (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, MUTED)):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
