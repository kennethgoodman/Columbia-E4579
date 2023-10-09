from src.recommendation_system.recommendation_flow.controllers.AbstractController import (
    AbstractController,
)
from src.recommendation_system.recommendation_flow.controllers.fall_2023 import (
    AlphaController,
    BetaController,
    CharlieController,
    DeltaController,
    EchoController,
    FoxtrotController,
    GolfController
)
from random import choice
from src.recommendation_system.recommendation_flow.retriever import ControllerEnum

controllers = [
    AlphaController,
    BetaController,
    CharlieController,
    DeltaController,
    EchoController,
    FoxtrotController,
    GolfController
]

class EngagementAssignmentController(AbstractController):
    def get_content_ids(self, user_id, limit, offset, seed, starting_point):
        controller = choice(controllers) # pick one randomly each time
        return controller().get_content_ids(user_id, limit, offset, seed, starting_point), controller
