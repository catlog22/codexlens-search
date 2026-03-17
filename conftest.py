import sys
import os

# Ensure the local src directory takes precedence over any installed codexlens_search package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
