"""
GazeAnywhere Inference Configuration
Simplified config for inference only - no training dependencies
"""
from detectron2.config import LazyCall as L
import sys
from os.path import dirname, abspath

# Add parent directory to path for imports
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from modeling import backbone, models, criterion

# Model configuration
model = L(models.AnyGazeModelMapper)()
model.backbone = L(backbone.build_backbone_dinov3txt)(
    name="dinov3_large"
)
model.tokenizer = L(backbone.build_tokenizer_dinov3txt)()
model.criterion = L(criterion.AnyGazeMapperCriterion)()
model.criterion.use_focal_loss = True
model.device = "cuda"
model.freeze_backbone = True
model.inout = True
model.patch_size = 16
model.dim = 512
model.num_layers = 6
model.image_size = 512
