from dataclasses import dataclass
from constants import DataDir

dataFiles = DataDir
miscDir = "misc/"

@dataclass(frozen=True)
class images:
    @dataclass(frozen=True)
    class directory:
        images = "img/"
        distributionAnalysis = images + "distributionAnalysis/"

@dataclass(frozen=True)
class keras_model_paths:
    @dataclass(frozen=True)
    class directory:
        kerasModels = "kerasModels/"
        autoencoderModels = kerasModels + "autoencoder/"

    @dataclass(frozen=True)
    class files:
        pass

    @dataclass(frozen=True)
    class extensions:
        kerasModel = ".km"
