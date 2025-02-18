import torch

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, finetuned_state_dict=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            with torch.no_grad():
                assert pretrained_checkpoint
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()

                if finetuned_state_dict:
                    print(f"Creating task vector from finetuned_state_dict based on {pretrained_checkpoint=}")
                elif finetuned_checkpoint:
                    print(f"Creating task vector from {finetuned_checkpoint=} based on {pretrained_checkpoint=}")
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()

                self.vector = {}
                # print(pretrained_state_dict.keys())
                # print(finetuned_state_dict.keys())
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                # Move tensors to the appropriate device
                pretrained_state_dict = {key: value.to(device) for key, value in pretrained_state_dict.items()}

                # 将 finetuned_state_dict 中的每个 tensor 移动到正确的设备
                finetuned_state_dict = {key: value.to(device) for key, value in finetuned_state_dict.items()}

                for key in pretrained_state_dict:
                    # print(pretrained_state_dict[key].dtype)
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        print(f"Key {key} has dtype {pretrained_state_dict[key].dtype} -- skipping!")
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def __truediv__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] / other
        return TaskVector(vector=new_vector)

    def __mul__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * other
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            #这里需要把pretrained_model的参数放到GPU上面
            new_state_dict = {}
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pretrained_state_dict = pretrained_model.state_dict()
            pretrained_state_dict = {key: value.to(device) for key, value in pretrained_state_dict.items()}
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
    
def merge_max_abs(task_vectors):
    """Mix multiple task vectors together by highest parameter value."""
    if len(task_vectors) == 0:
        return task_vectors[0]

    with torch.no_grad():
        new_vector = {}

        # Iterate over keys in the first task vector
        for key in task_vectors[0].vector:
            # Get the initial tensor for the current key
            max_abs_tensor = task_vectors[0].vector[key]

            # Iterate over the remaining task vectors
            for task_vector in task_vectors[1:]:
                if task_vector is None:
                    continue
                current_tensor = task_vector.vector[key]

                # Update max_abs_tensor to keep the element-wise maximum absolute values
                max_abs_tensor = torch.where(current_tensor.abs() >= max_abs_tensor.abs(), current_tensor, max_abs_tensor)

            # Assign the final tensor to the new_vector dictionary
            new_vector[key] = max_abs_tensor

    return TaskVector(vector=new_vector)


