import numpy as np
import sapien.core as sapien
import transforms3d
from gensim2.env.base.base_task import GenSimBaseTask
from gensim2.env.utils.pose import set_default_pose, set_random_pose


class LiftBanana(GenSimBaseTask):
    """Lift a tipped-over banana into an upright position."""

    def __init__(self, env, asset_id=""):
        # Define the task description and success criteria
        self.task_description = "Lift a banana into an upright position."
        self.sub_tasks = ["LiftBanana"]
        self.sub_task_descriptions = [
            "Lift a banana into an upright position"
        ]
        # Use a dummy success criterion to ensure progress array is properly sized
        # The actual progress checking is done in get_progress_state()
        self.success_criteria = ["dummy_criterion"]

        # For articulated tasks, articulator should be specified, while others should be None.
        super().__init__(env, rigid_body="banana", asset_id=asset_id)

    def reset(self, random=False):
        # Reset the environment to the initial state
        super().reset()

        if not random:
            set_default_pose(self.rigid_body, self.rigid_body_name)
        else:
            set_random_pose(
                self.rigid_body, 
                self.rigid_body_name, 
                task=self.task_type
            )

        # Return the initial observation of the environment
        return self.env.get_observation()

    def get_progress_state(self):
        """Custom success criterion for LiftBanana: check if banana is lifted above target height"""
        try:
            if self.rigid_body is not None:
                # Get the current height of the banana
                if isinstance(self.rigid_body, list):
                    # Handle multiple rigid bodies - check if any are lifted
                    heights = [rigid.pos[2] for rigid in self.rigid_body]
                    current_height = max(heights)
                else:
                    # Single rigid body
                    current_height = self.rigid_body.pos[2]
                
                # Define success criteria - more demanding lift
                target_height = 0.15  # 15cm above table for a proper lift
                threshold = 0.03      # 3cm tolerance (stricter)
                
                # Check if banana is lifted to target height
                if current_height > target_height - threshold:
                    return 1.0  # Task successful
                else:
                    # Return progress as fraction of target height
                    # progress = min(current_height / target_height, 1.0)
                    # return progress
                    return 0.0
            else:
                return 0.0
        except Exception as e:
            print(f"Error in LiftBanana progress check: {e}")
            return 0.0
