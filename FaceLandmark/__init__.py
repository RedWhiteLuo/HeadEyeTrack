import os
import sys

package_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(package_path)

from FaceLandmark.core.facer import FaceAna

__all__ = ['FaceAna']
