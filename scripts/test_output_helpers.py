from .output_helpers import *
def test_get_parent_classes():
    resnet18 = models.resnet18(pretrained=False)
    print(oh.get_parent_classes(resnet18))