from transformers import PreTrainedModel, PreTrainedTokenizer

class HiddenPatch:
    def __init__(
        self,
        model: PreTrainedModel,
        hiddens,
        layer_idx: int = 15,
        position_mask = None,
    ):
        self._model = model
        self._hooks = []
        self.hiddens = hiddens
        self.position_mask = position_mask
        self.layer_idx = layer_idx

    def __enter__(self):
        self._register_hooks()
        return

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _register_hooks(self):
        def patch_hook(layer_idx):
            def patch(mod, inp):
                if layer_idx == self.layer_idx:
                    if self.position_mask is not None:
                        inp0 = inp[0]+self.hiddens*self.position_mask
                    else:
                        inp0 = inp[0]+self.hiddens

                    return (inp0, *inp[1:])

            return patch

        for i, layer in enumerate(self._model.model.layers):
            hook = layer.register_forward_pre_hook(patch_hook(i))
            self._hooks.append(hook)


def ablation(hiddens, direction, rate = 1.0):
    weight = (hiddens*direction).sum(dim=-1, keepdim=True)
    hiddens = hiddens - rate*weight*direction
    return hiddens

def get_attn_layers(model):
    return [layer.self_attn for layer in model.model.layers]

def get_mlp_layers(model):
    return [layer.mlp for layer in model.model.layers]

class HiddenAblation:
    def __init__(
        self,
        model: PreTrainedModel,
        direction,
        rate = 1.0,
        # layer_idx: int = 15,
        position_mask = None,
    ):
        self._model = model
        self._hooks = []
        self.direction = direction/(direction.norm(dim=-1, keepdim=True)+1e-12)
        self.position_mask = position_mask
        self.rate = rate
        # self.layer_idx = layer_idx

    def __enter__(self):
        self._register_hooks()
        return

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _register_hooks(self):
        def abl_pre_hook(layer_idx):
            def abl_pre(mod, inp):
                if type(inp) == tuple:
                    inp0 = ablation(inp[0], self.direction, self.rate)
                    return (inp0, *inp[1:])
                else:
                    inp0 = ablation(inp, self.direction, self.rate)
                    return inp0
            return abl_pre
        
        def abl_hook(layer_idx):
            def abl(mod, inp, out):
                if type(out) == tuple:
                    out0 = ablation(out[0], self.direction, self.rate)
                    return (out0, *out[1:])
                else:
                    out0 = ablation(out, self.direction, self.rate)
                    return out0
            return abl
        
        for i, layer in enumerate(self._model.model.layers):
            hook = layer.register_forward_pre_hook(abl_pre_hook(i))
            self._hooks.append(hook)

        for i, layer in enumerate(get_attn_layers(self._model)):
            hook = layer.register_forward_hook(abl_hook(i))
            self._hooks.append(hook)
        
        for i, layer in enumerate(get_mlp_layers(self._model)):
            hook = layer.register_forward_hook(abl_hook(i))
            self._hooks.append(hook)