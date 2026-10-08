import sys

from PyQt6.QtWidgets import QApplication

from database.db import create_db

from view.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Inz Range")

    create_db()

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
