"""
... Then build Modules:

Linear
LayerNorm
Embedding
FeedForward
MultiHeadAttention
TransformerBlock
GPT


# Sample Linear layer to see how to apply nn.Module
class MyLinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.biases = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, inputs):
        return wx_plus_b.apply(inputs, self.weights, self.biases)

Notes:
    Don't use @dataclass for nn.Modules (while learning pytorch) because order matter on when super init is called and the attributes are defined


"""
