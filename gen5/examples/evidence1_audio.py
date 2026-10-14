"""Run the frozen Evidence-1 audio confirmation or a synthetic smoke check."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ammc_gen5.evidence1_audio import main

if __name__ == "__main__":
    main()
