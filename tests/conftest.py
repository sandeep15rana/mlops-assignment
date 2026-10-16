import os
import sys

# Ensure root directory is in sys.path for all pytest executions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
