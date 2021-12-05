from dataclasses import dataclass


@dataclass(frozen=True)
class keras_model_paths:
    @dataclass(frozen=True)
    class directory:
        kerasModels = "kerasModels/"
        aeModels = kerasModels + "autoencoder/"

    @dataclass(frozen=True)
    class files:
        pass

    @dataclass(frozen=True)
    class extensions:
        kerasModel = ".km"
