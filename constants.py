from dataclasses import dataclass, field


@dataclass
class TrainingConstants:
    R3NTrain: int = 450
    R2NTrain: int = 2760
    R3NValid: int = 50
    
    ftFactor: dict = field(default_factory=lambda:{1: 100, 2: 1000, 3: 100})
    trainFactor: dict = field(default_factory=lambda: {1: 100, 2: 1000, 3: 100})


@dataclass
class TestConstants:
    factor: int = 100
    SNum: int = 20000
    sNumR1: int = 0
    SNumTest: int = 0
    AcLen: int = 500
    
    angle: dict = field(default_factory=lambda: {1: -120, 2: 120, 3: 0})
    acIndices: dict = field(default_factory=lambda: {
                                                        1: [0, 500],
                                                        2: [500, 1000],
                                                        3: [1000, 1500],
                                                    })