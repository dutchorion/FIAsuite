"""Entry point used by the portable build (double-click FIASuite.exe)."""
import sys

from fiasuite.ui.main import run

if __name__ == "__main__":
    sys.exit(run(sys.argv))
